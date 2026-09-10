# baseline_session 修正实验方案

## 目标

只保留 `baseline_session`。实验输出两份独立结果：

- backend 结果：验证 session 复用、history 累积、resident/global KV、预算命中、restore、recompute、evict 和 queue。
- 风险结果：验证混合 session 中的 QA、Summary 后续 turn 能在 KIVI 与 H2O 下产生可归因的质量差异。

两份结果分开判定。风险不足时，backend 实验仍输出；policy 的优劣结论标记为“风险证据不足”，不能作为正式比较。

原方案要求 baseline_session 同时具备“会话结构 + 足够的 profile 风险信号”，并要求 backend 事件链与 KIVI/H2O 信号。这里把这两个要求拆成两个 gate，不再让其中一个阻断另一个。

## 数据集

使用 48 个五轮 session，共 240 个请求。原方案的“24 到 36 个 session”与“eval 至少 8 个 KIVI、8 个 H2O、8 个对照 session”及 50/50 split 不兼容：eval 至少需要 24 个 session，因此总量至少应为 48。

每个风险组各 16 个 session，8 个 calibration、8 个 eval：

| 组别 | QA | Summary | 用途 |
| --- | ---: | ---: | --- |
| `kivi_sensitive` | 8 | 8 | KIVI 主风险组 |
| `h2o_sensitive` | 8 | 8 | H2O 主风险组 |
| `low_risk` / `tie_sensitive` | 8 | 8 | 对照组 |

每个 session 固定五轮：

1. `turn0`、`turn1` 保留结构合格的 ShareGPT 对话。
2. `turn2` 注入严格筛出的 LongBench QA 或 Summary 内容。
3. `turn3` 用固定的人工模板要求复述 `turn2` 的结论，参考答案复用 LongBench 的参考答案。
4. `turn4` 用第二个固定模板要求在不引入新事实的前提下重述上一轮结论，参考答案仍为同一 LongBench 参考答案。

QA turn 保持 `task=qa`，以 F1 计算质量差异；Summary turn 保持 `task=summary`，以 ROUGE-L 计算。ShareGPT 开场 turn 保持 `chat`，只用于 session/backend 语义，不进入风险质量统计。

严格风险池规则：

- KIVI：`kivi_max >= 0.05` 且 `kivi_max - h2o_max >= 0.02`。
- H2O：`h2o_max >= 0.05` 且 `h2o_max - kivi_max >= 0.02`。
- 对照：所有主流 profile 的 `quality_loss <= 0.01`。
- Tie：两家族均高但差值不足，不归入单家族风险组，只能进入对照组。

扩大 LongBench 候选与实测范围，直到每个严格风险组都有足够内容。不得用 tie 样本补成 KIVI 或 H2O 主风险组。

## 构造与契约

外部标注工作区新增 hybrid session 构造器，替代当前从既有 ShareGPT 混合源直接筛选的方式。

每条注入 turn 必须保留：

- `task`、`reference`、`session_id`、`turn_index`、`arrival_index`。
- `source = hybrid_session_builder`。
- `source_dataset = sharegpt_longbench_hybrid_session`。
- `content_source_dataset`、`content_source_request_id`、`content_source_index`。
- `risk_family`、`injection_template`、`original_session_id`。

候选先按 session 结构、五轮完整性和 effective prompt 上限筛骨架，再按 LongBench 实测风险分配内容。不得按最短 prompt 排序。split 在 session 级完成，保证每个风险组和任务在 calibration/eval 两侧各有 4 个 session；随后交错分配全局 `arrival_index`。

导出前重新跑 hybrid candidates 的 measured profile。session 风险标签只能来自这次混合后的测量，不能直接沿用 LongBench 单轮标签。

## Gate 与输出

`trace_semantics_gate` 必须通过：

- fixture 的 session、turn、arrival 契约合法。
- profile 表含 resident KV 与 history 字段。
- replay 中出现 session reuse、global resident 演化和至少一个真实压力事件。

`risk_signal_gate` 检查 eval：

- KIVI 与 H2O 各至少一个主流 profile 的组内平均质量损失超过 `0.02`。
- 两个风险组都覆盖 QA 和 Summary。
- 敏感组相对对照组有正向风险差。
- 每项质量记录可追溯到注入内容与模板。

两个 gate 都写入独立 JSON。`trace_semantics_gate` 通过后始终执行 backend policy replay；`risk_signal_gate` 失败时，summary 保留 backend 结果，并将 policy 比较状态写为 `risk_evidence_insufficient`。

## 测试

- 候选构造测试：48 个完整五轮 session、注入 metadata、任务语义、连续 turn 和交错 arrival。
- 风险池测试：严格 KIVI/H2O、对照、tie 的边界值及扩池不足时报错。
- split 测试：每个风险组、QA、Summary 在 calibration/eval 的 session 数量固定。
- gate 测试：backend gate 与风险 gate 独立；风险失败不阻断 backend replay；policy 结果被正确标记为不可正式比较。
- 回归测试：chat 注入内容不能进入风险统计；缺少 LongBench provenance 的 fixture 被拒绝。

## 假设

继续使用当前 Qwen2.5-7B-Instruct、现有主流 KIVI/H2O profile 和 240 请求上限。不再维护 `baseline_quality` 相关代码、夹具或文档。

## 当前执行进度（2026-09-01）

这轮方案已经做完前三块，卡在第 4 块的真实测量。

主仓库里已经落地的改动可以直接沿用：

- `scripts/import_external_fixtures.py` 已经收紧 `baseline_session` 的 fixture 契约，要求固定 `48 x 5` 的 hybrid session 结构，并校验 provenance、`request_id` 唯一性、turn/role 合法性。
- `scripts/validate_trace_quality.py` 已经把风险 gate 拆出来，只统计 eval 的 QA/Summary 注入 turn，排除 calibration、chat 和非法 provenance，并补了 group mean、control gap、provenance evidence。
- `run_util/experiment.py` 已经改成 `trace_semantics_gate` 是 replay 的唯一硬门槛。trace 失败时风险状态写 `not_evaluated`，不跑 replay；trace 通过后先跑 policy replay，再看风险 gate。风险不足时 backend 结果保留，状态写 `risk_evidence_insufficient`。
- `run_util/experiment_summary.py` 已经补上 `policy_comparison_status` 的传递，普通 summary、policy metrics 和 total summary 都能看到。
- 这部分对应的提交已经在 `main` 上：`ba24899`、`ebd9127`、`bded933`、`5f79bad`、`ff9f2e0`。

外部标注工作区 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling` 也已经做完候选构造和测试：

- 新增 `scripts/build_hybrid_sessions.py`、`tests/test_hybrid_sessions.py`，并修改了 `scripts/label_sharegpt_sessions.py`。
- 已生成 `48` 个五轮 session，共 `240` 条 candidate，请求文件在 `artifacts/candidates/hybrid_session_candidates.jsonl`。
- 当前规则仍是严格风险池，不允许拿 tie 样本去补 `kivi_sensitive` 或 `h2o_sensitive`。

已经确认过的验证结果：

- 外部工作区测试 `10 passed`。
- 主仓库这次相关测试里，聚焦的 session/config/import 回归是 `41 passed`。
- 全量 `pytest` 是 `342 passed, 3 failed`。这 3 个失败是既有遗留问题，不是这次新增逻辑直接引入的：
  - 两个 legacy ShareGPT fixture 测试还在吃旧 schema。
  - 一个 `pilot_session_trace` 的 profile 顺序断言和本次修正无关。

## 当前阻塞点

真实 profile 测量还没跑完，所以正式 fixture 还没导出，也还没导入主仓库，更没跑最终 smoke。

已经试过两轮：

- 第一轮直接跑全量 `240` 条 candidate，`full_gpu` 在 `hybrid-session-009-turn-2` 附近 OOM。
- 第二轮把 `CUDA_VISIBLE_DEVICES` 固定为 `0,1`，并把 `profile_chunk_size` 降到 `8`、`vllm_gpu_memory_utilization` 降到 `0.65`，还是在 `full_gpu` 上 OOM。
- 第三轮继续收紧到 `profile_chunk_size: 4`、`use_persistent_workers: false`、`vllm_gpu_memory_utilization: 0.55`，并加上 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。全量跑时，失败点后移到了 `hybrid-session-019-turn-3`，但还是没跑穿。
- 之后又把 candidate 按 session 一分为二，生成了：
  - `artifacts/candidates/hybrid_session_candidates_part1.jsonl`
  - `artifacts/candidates/hybrid_session_candidates_part2.jsonl`
  - `configs/hybrid_session_candidates_part1.yaml`
  - `configs/hybrid_session_candidates_part2.yaml`
  前半批 `part1` 仍然在 `hybrid-session-019-turn-3` 这一带失败。

到目前为止，最可靠的结论只有一个：问题不在风险 gate，也不在导入脚本，而是在 `full_gpu` 真实测量这一步的显存上限。

相关文件现在都在：

- 测量输出：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_profiles.csv`
- 失败诊断：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_profiles_failed_chunks.csv`
- 前半批输出：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_part1_profiles.csv`
- 前半批失败诊断：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_part1_profiles_failed_chunks.csv`

## 其他会话接手时该怎么继续

先别动主仓库逻辑，也别改风险阈值。先把测量跑通，再说后面的 label、导入和 smoke。

建议按下面的顺序继续：

1. 先读一眼失败诊断，确认最新失败点还是 `hybrid-session-019-turn-3`，而不是更后面的请求。
2. 继续沿用双卡 `0,1`。这一步已经验证过确实在吃两张卡，不是单卡误配。
3. 优先继续缩小执行粒度，而不是改判定标准。可以再把 candidate 拆得更细，比如按更小的 session 批次跑，再合并 CSV。
4. 如果需要改临时配置，只改外部工作区 `configs/hybrid_session_candidates*.yaml`，不要顺手改主仓库正式配置。
5. 一旦 measured profile 跑齐，立刻执行：

```bash
cd /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling
python scripts/label_sharegpt_sessions.py \
  --candidates artifacts/candidates/hybrid_session_candidates.jsonl \
  --measurements artifacts/measurements/hybrid_session_candidates_profiles.csv \
  --output artifacts/fixtures/baseline_session_external.jsonl \
  --manifest artifacts/manifests/baseline_session_external_manifest.json
```

6. 再回主仓库做 validate-only 导入：

```bash
cd /DATACENTER3/zhenxiang.wang/work/TailGuardKV
python scripts/import_external_fixtures.py \
  --kind baseline_session \
  --input /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_session_external.jsonl \
  --validate-only
```

7. validate-only 过了以后，再跑：

```bash
cd /DATACENTER3/zhenxiang.wang/work/TailGuardKV
bash scripts/run_external_baseline_smoke.sh session
```

8. 最后检查四样东西：
   - `trace_semantics_gate` JSON
   - `risk_signal_gate` JSON
   - summary 里的 `policy_comparison_status`
   - backend replay 是否在风险不足时依然保留

## 接手时别踩的坑

- 当前 `main` 里还有用户自己的 dirty changes，不要 `reset`、`checkout --`、`clean`。
- `baseline_quality` 相关代码和夹具不要顺手删。这次的约束是“不再扩展”，不是“立刻清掉”。
- 外部标注工作区不是 Git 仓库，那里没有提交记录，改动要靠文件本身和生成物追踪。
- 在真实测量没跑完前，不要声称这个方案已经闭环。

## 运行环境说明

- 相关脚本统一使用 `conda run -n tailguardkv-base ...` 运行，不要切到别的环境。
- 真实 profile 测量默认只允许用 `CUDA_VISIBLE_DEVICES=0,1`，这两张卡已经验证过可用。
- 长时间后台任务建议用 `nohup` 加 `setsid` 启动，避免终端断开后进程一起退出。
- 临时调参只改外部标注工作区里的临时配置，不动主仓库正式配置，除非先确认要把参数固化回主线。

## 当前执行进度（2026-09-02 更新）

以下状态覆盖上一节中关于候选构造、测试和下一轮测量粒度的描述；上一节保留为 2026-09-01 的历史记录。

### 已完成

- 已修正 hybrid 构造器的默认 ShareGPT 输入，现使用 `artifacts/candidates/sharegpt_hybrid_skeletons.jsonl`，不再误用只有 36 个 session 的 `sharegpt_session_candidates.jsonl`。
- 已为注入的 LongBench `turn2` prompt 增加 `max_content_prompt_chars`，默认上限为 `512`；构造 manifest 写入该参数，注入 metadata 写入 `content_prompt_chars`。ShareGPT 的 `turn0`、`turn1` 和固定复述的 `turn3`、`turn4` 未被截断或改变语义。
- 已重建 `artifacts/candidates/hybrid_session_candidates.jsonl`：`48` 个 session、每个 `5` 轮、共 `240` 条请求；所有注入内容 prompt 长度不超过 `512`。
- 已增加注入 prompt 截断的回归测试。外部标注工作区的 hybrid 构造与标注测试最新结果为 `11 passed`。
- 已生成 4 个独立测量批次，每批 `12` 个 session、`60` 条请求：`hybrid_session_candidates_batch01.jsonl` 至 `batch04.jsonl`，以及对应 YAML 配置。

### 当前测量状态

- 全量历史测量 `artifacts/measurements/hybrid_session_candidates_profiles.csv` 仍是旧的中断结果：只含 `full_gpu` 的部分记录，不能作为风险标注输入，也不能与新候选混用。
- 四个新批次尚未开始真实 profile 测量。目前没有 `build_profile_table`、label 或 external smoke 进程运行。
- 新批次均固定使用双卡 `CUDA_VISIBLE_DEVICES=0,1`，并采用已准备好的保守参数：`profile_chunk_size: 1`、`use_persistent_workers: false`、`vllm_gpu_memory_utilization: 0.55`、`vllm_max_model_len: 1024`、`vllm_enforce_eager: true`。这次缩小的是单次独立测量的 session 批次和内容长度，未修改风险阈值、profile 列表或主仓库 gate 逻辑。

### 未完成的验收链路

1. 依次执行 4 个 batch 的真实测量；每个 batch 必须覆盖 `full_gpu`、全部 KIVI 和全部 H2O profile，且所有请求测量成功。
2. 合并 4 个 batch CSV，形成只含本轮 `48 x 5` hybrid candidates 的完整测量表。
3. 用完整测量表运行 `label_sharegpt_sessions.py`，导出正式 `baseline_session_external.jsonl` 及 manifest；严格 KIVI/H2O 池不足时应报错，不能以 tie 样本补足。
4. 在主仓库完成 `import_external_fixtures.py --validate-only`，随后运行 `scripts/run_external_baseline_smoke.sh session`。
5. 按《理想baseline_smoke表说明.md》核验 `trace_semantics_gate`、`risk_signal_gate`、`policy_comparison_status`，以及风险证据不足时 backend replay 仍被保留的行为。

### 下一步命令

以 batch01 为例，其他 batch 仅替换编号和输出文件名：

```bash
cd /DATACENTER3/zhenxiang.wang/work/TailGuardKV
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n tailguardkv-base python -m run_util.build_profile_table \
  --config ../TailGuardKV-labeling/configs/hybrid_session_candidates_batch01.yaml \
  --output ../TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_batch01_profiles.csv \
  --no-dry-run
```

在所有批次完整成功前，`baseline_session_external.jsonl` 和其 manifest 仍是 2026-08-13 的旧产物，不可作为本方案的正式验收数据。

## 当前执行进度（2026-09-02 续）

- batch01 的首次启动没有进入 GPU 测量：`profile_chunk_size: 1` 与
  `use_persistent_workers: false` 会走单请求路径，绕过 session runtime，导致
  `resident_kv_mib_before`、`resident_kv_mib_after` 和 `kv_cumulative_mib` 缺失，
  被 baseline-session profile 契约拒绝。
- 已用 13 请求、8 个 profile 的持久 worker probe 验证最小修正：104 条记录全部
  `ok/measured`，resident 字段完整，且运行时记录确认 `CUDA_VISIBLE_DEVICES=0,1`
  与 `balanced_two_gpu` 生效。
- 因此仅修改外部标注工作区四个临时 batch 配置的
  `use_persistent_workers: true`；未改风险阈值、profile 列表、内容长度、显存利用率
  或主仓库正式配置。
- batch01 已完整通过：共 480 条记录，8 个 profile 各 60 条，全部
  `ok/measured`，resident 字段无缺失，且未生成失败 chunk。GPU 已释放，后续 batch
  继续严格串行运行。
- batch02 已按同一验收标准完整通过：共 480 条记录，8 个 profile 各 60 条，全部
  `ok/measured`，resident 字段无缺失，且未生成失败 chunk。batch03 是下一批待运行
  测量。
- batch03 已按同一验收标准完整通过：共 480 条记录，8 个 profile 各 60 条，全部
  `ok/measured`，resident 字段无缺失，且未生成失败 chunk。batch04 是最后一批待运行
  测量。
- batch04 已按同一验收标准完整通过。至此四个 batch 的测量均完成；已合并为仅含本轮
  `48 x 5` hybrid candidates 的正式测量表，共 1,920 条唯一 request/profile 记录，全部
  `ok/measured`，resident 字段无缺失。
- 严格风险标注已实际执行并按预期拒绝不足池：`kivi_sensitive qa hybrid session pool
  insufficient: 0 < 8`。汇总的实际分布为 `qa/h2o=3`、`summary/kivi=2`、
  `qa/kivi=0`，其余主要为 low/tie；这不是 CSV 合并或 provenance 错误。不得用 tie
  样本补齐，因此正式 fixture/manifest、主仓库 validate-only 和 external smoke 尚未执行。
- 下一轮必须扩大 LongBench 候选与实测范围，重新构造并测量足以满足每个严格风险组和
  QA/Summary 各 8 个 session 的候选池；不能放宽阈值或改变风险组定义。
