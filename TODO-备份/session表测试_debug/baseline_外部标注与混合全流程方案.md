# 外部标注、数据集混合与 Baseline 全流程方案

> **实施提示：** 按任务逐项执行；所有真实 profile、标注与 smoke 程序必须异步启动，结束或报错时唤醒会话。GPU 测量严格串行，不轮询运行中的进程。

> **收尾与资产保留：** 全流程完成后清理 `out/`，将最后两个通过全部 gate 的有效结果以及 baseline smoke 结果归档到 `out-备份/`。`TailGuardKV-external-baseline`、`TailGuardKV-labeling` 等隔离或冗余目录中的关键代码、测试、脚本和配置必须迁入主仓库并提交到 GitHub，归档后方可删除冗余目录。

## 执行环境与异步启动

- 外部候选构造、标注、校验、导入和主仓库 runner 的入口统一使用 Conda 环境 `tailguardkv-base`：`conda run -n tailguardkv-base ...`。不要依赖交互式 `conda activate`。
- KIVI、H2O 与 vLLM 的专用依赖环境由现有 profile adapter/subprocess 按项目配置调度；不要绕过 runner 手动替换其环境。
- 长时间程序必须以 `nohup + setsid` 异步启动并记录日志与 PID。命令名是 `nohup` 和 `setsid`，不是 `nohub` 或 `setid`。标准模板如下：

  ```bash
  nohup setsid env CUDA_VISIBLE_DEVICES=0,1 \
    conda run -n tailguardkv-base python -m run_util.build_profile_table \
    --config <config.yaml> --output <profiles.csv> --no-dry-run \
    > <run.log> 2>&1 < /dev/null &
  echo $! > <run.pid>
  ```

- 一个 GPU 任务退出并写完结果或错误日志后，才允许启动下一任务；会话由进程结束或报错唤醒，不以轮询代替完成信号。

## 目标

构造并外部标注可正式使用的 `baseline_session` 与 `baseline_quality` 数据集，随后完成 measured smoke。最终产物必须得到可解释、可正式比较的 baseline 结果，而非仅生成候选、CSV 或一次可运行实验。

## 总体设计

- ShareGPT 仅提供真实的两轮会话骨架，保留 session/history/prefix 复用语义。
- LongBench 是 QA 与 Summary 的主内容来源；Summary 只能来自 LongBench。
- RAGhot_QA 仅作为 QA 后备来源：LongBench 扩池仍无法补齐严格 QA 风险池时才启用。
- 所有最终标签只来自完整最终形态的输入在 `full_gpu`、4 个 KIVI 和 3 个 H2O profile 上的实测；单轮预筛结果只用于排序，不能继承为最终标签。
- 两张 baseline 均采用严格风险归因：KIVI/H2O 敏感阈值为 `0.05`，跨家族差至少为 `0.02`，low-risk 为所有主流 profile 损失不高于 `0.01`；tie 一律拒绝，不能回填任何敏感组。

## 全局约束

- 模型固定为 Qwen2.5-7B-Instruct，profile 固定为 `full_gpu`、4 个 KIVI 和 3 个 H2O。
- 外部 session 测量固定双卡 `CUDA_VISIBLE_DEVICES=0,1`、`profile_chunk_size: 1`、`use_persistent_workers: true`、`vllm_gpu_memory_utilization: 0.55`、`vllm_max_model_len: 1024`。
- 每个长时命令通过现有异步 launcher 或 `nohup`/`setsid` 启动，保留 PID 和日志；不得并行占用两张 GPU。
- `baseline_session` 最终固定为 48 个 session、每个 5 轮、共 240 条请求；每个 `risk_family x task x split` 恰有 4 个 session。
- 未满足任一数据 gate 时，禁止导入正式 fixture、禁止宣布 policy 优劣、禁止将结果作为 TailGuard/Oracle 参照。

---

## 任务 1：建立来源无关的 provenance 与 gate

**文件：**

- 修改 `scripts/import_external_fixtures.py`
- 修改 `scripts/validate_trace_quality.py`
- 修改 `run_util/experiment.py`
- 修改 `tests/test_external_fixture_import.py`
- 新增 `tests/test_baseline_quality_gate.py`

### 交付内容

1. 将当前硬编码的 `sharegpt_longbench_hybrid_session`、`longbench` 和 `longbench_content` 校验改为来源注册表。
2. 会话来源允许 `longbench` 与 `raghot_qa`；顶层构造来源保持 `hybrid_session_builder`。
3. 五轮角色固定为：

   - `turn0`/`turn1`: `sharegpt_opening`
   - `turn2`: `content_query`
   - `turn3`: `reference_recall`
   - `turn4`: `reference_rewrite`

4. 所有注入 turn 必须包含：

   ```text
   content_source_dataset
   content_source_request_id
   content_source_index
   content_payload_hash
   injection_template
   original_session_id
   hybrid_turn_role
   ```

5. RAGhot 额外要求：`context_pack_hash`、`supporting_fact_ids`、`packing_policy_version`。
6. 新增 `baseline_quality_signal_gate` JSON 输出，检查 180 条 strict fixture、三组各 60 条、calibration/eval 覆盖、8 profile 测量完整性、敏感组相对 low-risk 的正向风险差，以及 KIVI/H2O 各至少一个 profile 的组均值 `quality_loss > 0.02`。
7. 仅在相应 signal gate 通过后写入 `policy_comparison_status=formally_comparable`；未通过时为 `risk_evidence_insufficient`。

### 测试

- LongBench 与 RAGhot 的合法 provenance 均可通过。
- 缺少 RAGhot context/evidence 字段、来源被伪造、role 与 turn 不一致、profile 不全、tie 被重标，均必须失败。
- quality gate 失败时，policy 输出保留但状态为 `risk_evidence_insufficient`。

---

## 任务 2：构造可验证的 RAGhot QA 候选

**文件：**

- 新增 `../TailGuardKV-labeling/scripts/prepare_raghot_qa_candidates.py`
- 新增 `../TailGuardKV-labeling/tests/test_raghot_candidates.py`
- 修改 `../TailGuardKV-labeling/scripts/common.py`

### 输入与筛选

输入为 `/DATACENTER3/zhenxiang.wang/resource/RAGhot_QA/validation-00000-of-00001.parquet`。

1. 读取 `id`、`question`、`answer`、`context`、`supporting_facts`。
2. 拒绝空 ID、问题或答案为空的记录。
3. 对每一个 supporting fact，必须在相同 title 的 context sentence 中定位到其 `sent_id`；任一事实无法定位即拒绝。当前实测只有 2,088 条满足全部证据可定位，不能将其余记录作为正式候选。
4. 先按原始文档/句子顺序放入全部 supporting sentences，再按相同顺序补充非 supporting sentence 作为干扰上下文。
5. 上下文总长度上限为 3,000 字符；所有 supporting sentence 放不下时拒绝，绝不从头部截断。
6. prompt 固定为 `Context + Question + Answer:`；reference 为原始 `answer`。

### 输出契约

每条候选包含稳定 request ID、`task=qa`、reference、来源记录 ID、完整 evidence ID 列表、context pack SHA-256、payload SHA-256 与 `packing_policy_version=raghot_support_first_v1`。

### 测试

- 完整证据样本的 pack 含全部事实且哈希稳定。
- 缺少 supporting title/sentence、支持句超限、空字段样本被拒绝。
- 非支持句只能在保留全部支持句后追加。

---

## 任务 3：扩展 LongBench 并完成直接预筛测量

**文件：**

- 修改 `../TailGuardKV-labeling/scripts/prepare_longbench_candidates.py`
- 修改 `../TailGuardKV-labeling/configs/longbench_labeling.yaml`
- 新增 `../TailGuardKV-labeling/scripts/partition_labeling_batches.py`
- 新增 `../TailGuardKV-labeling/tests/test_labeling_batches.py`

### 交付内容

1. 扫描所有可用 LongBench QA 与 Summary 行，不再固定取每类前 200 条。
2. 记录 LongBench 子集、原始 source ID、任务、reference、prompt hash 与候选顺序。
3. 使用确定性 60 请求批次；每个 batch manifest 列出输入哈希、8 个预期 profile 和对应 CSV 输出。
4. 异步、串行运行各批直接 profile 测量；只合并 request ID、profile、候选 hash 一致且 `ok=true`、`measured=true` 的行。
5. 根据严格直接风险结果优先选择后续 hybrid 内容。若不足则继续下一个 LongBench batch，直到达到 hybrid 预筛需要或 LongBench 来源耗尽。

### 测试

- 相同输入产生相同 batch 分片和 manifest。
- 合并器拒绝重复 request/profile、候选 hash 不同、失败 profile 或缺 profile 的 CSV。
- 直接标签只能出现在预筛 manifest 中，不能写入最终 fixture 风险字段。

---

## 任务 4：构造、测量和筛选混合 session 候选

**文件：**

- 修改 `../TailGuardKV-labeling/scripts/build_hybrid_sessions.py`
- 修改 `../TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`
- 新增 `../TailGuardKV-labeling/scripts/select_hybrid_candidate_batches.py`
- 修改 `../TailGuardKV-labeling/tests/test_hybrid_sessions.py`

### 构造规则

1. 每个候选使用 ShareGPT 两轮开场、一个 QA 或 Summary 内容 turn，以及两个固定复述 turn。
2. QA 先选择 LongBench；当所有 LongBench QA 批次完成而任一严格 QA cell 少于 12 个完整 hybrid 候选时，才构造 RAGhot QA batch。
3. Summary 只使用 LongBench；若任一 Summary cell 少于 12 个，不得用 RAGhot 或 tie 补齐，继续扩展 LongBench 或失败。
4. 候选探索阶段可以复用 ShareGPT skeleton；最终 fixture 中每个 `original_session_id`、内容 source record 和注入 payload 均必须唯一。
5. 每批 12 个 session、60 条请求，按已验证的 persistent-worker 双卡配置异步串行测量。

### 风险分类与停止条件

1. 使用 turn2--turn4 中每个 profile 的最大质量损失分类该 session。
2. 每个 `kivi_sensitive/h2o_sensitive/low_risk x qa/summary` cell 必须至少保留 12 个 profile 完整、严格合格的候选。
3. 不足时继续按预筛排序构造下一 batch；来源耗尽仍不足时输出不足报告并终止，不生成 fixture。

### 测试

- five-turn 完整性、全局 arrival 交错、session 内一致 provenance。
- RAGhot QA 可构造，RAGhot Summary 必须被拒绝。
- 最终选择拒绝重复 skeleton、重复内容、tie 和不完整 profile。
- 任何 cell 少于 12 时导出失败并包含确切 cell、可用数和来源耗尽状态。

---

## 任务 5：导出严格的 quality 与 session fixture

**文件：**

- 修改 `../TailGuardKV-labeling/scripts/label_longbench_quality.py`
- 修改 `../TailGuardKV-labeling/scripts/label_sharegpt_sessions.py`
- 新增 `../TailGuardKV-labeling/tests/test_strict_fixture_selection.py`

### baseline_quality

1. 删除 `tie_sensitive` 回填逻辑；tie 仅写入 rejected manifest。
2. 导出 `baseline_quality_external.jsonl`：严格 KIVI/H2O/low-risk 各 60 条，共 180 条，calibration/eval 各 90 条。
3. LongBench 是主来源；仅在 LongBench 严格 QA 样本不足以填满所需质量 risk family 时才使用已经测量的 RAGhot QA。
4. manifest 必须写入三组 source 分布、任务分布、被拒绝 tie、完整 8-profile 覆盖和 fixture hash。

### baseline_session

1. 导出 `baseline_session_external.jsonl`：48 个五轮 session、240 条记录。
2. 每个 `risk x task x split` 恰有 4 个 session；每个 `risk x task` 共 8 个 session。
3. 先按风险 margin 降序，再按 source request ID 升序；按序交替分给 calibration/eval。
4. manifest 必须写入候选 reserve、来源、原始 skeleton ID、内容 ID、payload hash 和每个 profile 覆盖。

### 测试

- `baseline_quality` 严格三组各 60、90/90 split。
- `baseline_session` 为 48x5、12 个 risk/task cell 各 4。
- tie、重复内容、重复最终 skeleton、缺 provenance、source hash 不一致必须失败。

---

## 任务 6：导入并运行两个正式 baseline smoke

**文件：**

- 修改 `../TailGuardKV-labeling/scripts/import_labeled_fixtures_into_repo.sh`
- 修改 `configs/pilot_external_baseline_quality.yaml`
- 修改 `configs/pilot_external_baseline_session.yaml`
- 修改 `scripts/run_external_baseline_smoke.sh`

### 执行顺序

1. 对 quality/session fixture 分别执行 `import_external_fixtures.py --validate-only`。
2. 校验 fixture SHA-256 与 manifest 中记录一致后，才导入 `data/fixtures/`。
3. quality smoke 先运行，完成并通过 quality signal gate 后再运行 session smoke。
4. 两个 smoke 均通过现有异步 measured runner 启动，并在各自带时间戳的目录保存 profile CSV、gate JSON、policy CSV、summary CSV、日志与 PID。
5. 同一时刻只能有一个 profile/smoke GPU 进程。

### 正式验收

#### baseline_quality

- 导入校验通过，180 条 fixture 严格分布正确。
- 8 个 profile 对所有请求完整 `ok/measured`。
- `baseline_quality_signal_gate.passed=true`。
- 五条 policy 都有结果，`policy_comparison_status=formally_comparable`。

#### baseline_session

- 导入校验通过，48x5 契约、交错 arrival、split/risk/task 分布正确。
- `trace_semantics_gate.passed=true`，并实际出现 session reuse、global resident 演化和至少一个真实压力事件。
- `risk_signal_gate.passed=true`；两敏感组均覆盖 QA 与 Summary，对应家族至少一个 profile 组均值损失大于 `0.02`，且相对 low-risk 有正向风险差。
- `policy_comparison_status=formally_comparable`。

#### 结果可解释性

- `full_lru` 维持 exact 行为；其余策略不能全部退化为 exact。
- dynamic policy 必须出现 lossy 行为，并与 memory 收益或质量/时延代价对应。
- 预算变松时，预算命中、restore/recompute/queue 不应系统性变差；`epsilon` 放松时 lossy 比例和质量损失应可见变化；更严格 `delta` 应使 `static_safe` 更保守。
- 若任一 gate 为 `risk_evidence_insufficient` 或 `not_evaluated`，只可保留诊断和 backend 数据，不得得出 policy 优劣结论。

## 提交节奏

1. `feat: 增加 RAGhot 候选与来源契约`
2. `feat: 严格混合会话筛选与标注`
3. `feat: 增加 baseline 质量信号门禁`
4. `test: 覆盖外部 fixture 严格验收`
5. `docs: 更新外部标注执行流程`

---

## 执行进度与续跑指南（2026-09-03）

### 当前状态

- 已完成并经独立复审：任务 1--4 的代码与测试实现。
- 未开始：任务 5 的 strict fixture 导出、任务 6 的导入与正式 GPU smoke。
- 未启动任何真实 profile/GPU 测量；因此当前不存在可用于论文结论的正式 baseline 结果，也不得宣布 policy 优劣。
- 任务 5 实现代理因服务余额不足而未启动；该任务尚未写入业务代码。

### 已完成的改动

1. 主仓库隔离分支 `feat/external-baseline-flow`（worktree: `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline`）已提交：
   - `e1901f1 feat: 增加 baseline 质量信号门禁`
   - `32d1d4b fix: 拒绝空 RAGhot 证据`
   - provenance 改为来源注册表，允许 LongBench 与 RAGhot QA；五轮角色固定为 `sharegpt_opening`、`sharegpt_opening`、`content_query`、`reference_recall`、`reference_rewrite`。
   - RAGhot 必需 context/evidence/packing provenance；空 evidence 也会在 experiment 的直接质量 gate 路径被拒绝。
   - `baseline_quality_signal_gate` 已接入 policy 状态：gate 不通过时保留 policy 输出，但状态为 `risk_evidence_insufficient`。

2. 已获授权直接修改的非 Git 标注目录 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling`：
   - 新增 RAGhot QA 候选构造器，验证同 title/sent_id 证据、support-first packing、3,000 字符硬上限、原始 answer/reference 保留、稳定 SHA-256 及 `raghot_support_first_v1`。
   - LongBench 候选改为扫描全部 QA/Summary，新增确定性 60-request prescreen batch、8-profile manifest、严格 CSV 合并、严格风险排序与按需调度。
   - 直接预筛标签被隔离在 prescreen manifest；旧的直接测量到最终 fixture 路径已禁用。异步 launcher 进入隔离 TailGuardKV worktree，并使用跨进程 `flock` 避免双卡任务重叠。
   - 混合 session 已支持 LongBench-first 和可验证的 RAGhot QA 回退；回退必须同时证明完整 LongBench QA batch 清单已完成且 final-form QA 严格候选仍不足。RAGhot Summary 被构造与最终选择双重拒绝。
   - 最终 session 候选校验要求完整 8-profile 覆盖、五轮角色/任务/reference/provenance 一致、原始 skeleton/内容 source record/payload 唯一、每个 risk/task cell 至少 12 个 reserve；不足报告包含确切 cell、可用数和真实 exhaustion 状态。

### 已验证

- 主仓库任务 1 聚焦测试 `17 passed`，相关回归 `27 passed`，修复后覆盖 `34 passed`；完整套件曾报告 `352 passed, 3 subtests passed`，另有 4 项既存且与本任务无关的失败，详见隔离账本的 task-1 report。
- 标注目录在任务 2、3、4 的最终验证分别为 `16 passed`、`27 passed`、`41 passed`；任务 4 最终独立复审聚焦测试为 `25 passed`。
- 所有已运行的测试、编译与数据准备程序按异步 `nohup setsid` 模式启动并保留 PID/日志；未轮询活跃进程。

### 下一会话执行顺序

1. 从任务 5 开始，先阅读本文件任务 5 与隔离账本：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-external-baseline/.superpowers/sdd/baseline_外部标注与混合全流程方案/progress.md`。
2. 在 `TailGuardKV-labeling` 完成 strict quality/session fixture exporter 与 `tests/test_strict_fixture_selection.py`，并先跑失败测试再实现。禁止恢复 `tie_sensitive` 回填，禁止把 direct prescreen risk 写入最终 fixture。
3. 在 fixture 尚未同时满足 strict 180-row quality 与 48x5 session reserve/coverage 契约前，不得导入 `data/fixtures/`，不得启动 smoke。
4. 任务 6 先更新 persistent-worker 配置：双卡 `CUDA_VISIBLE_DEVICES=0,1`、`profile_chunk_size: 1`、`use_persistent_workers: true`、`vllm_gpu_memory_utilization: 0.55`、`vllm_max_model_len: 1024`。任务 4 launcher 会刻意拒绝旧的 `use_persistent_workers: false`/`profile_chunk_size: 4` 配置。
5. 真实测量严格串行：先 quality，再确认 `baseline_quality_signal_gate.passed=true`，之后才运行 session。每个命令通过 `nohup + setsid + conda run -n tailguardkv-base` 启动，保存 PID 和日志；一个两卡任务结束并写完结果或错误日志后，才可启动下一个。
6. 导入前先分别执行 `import_external_fixtures.py --validate-only` 并核对 fixture SHA-256/manifest；任何 gate 为 `risk_evidence_insufficient` 或 `not_evaluated` 时，只保留诊断数据，不得形成 policy 比较结论。

---

## 运行账本更新（2026-09-03，后续会话以此节为准）

### 已落地的执行变更

- `TailGuardKV-labeling` 已完成任务 5 的 strict exporter；聚焦测试 `tests/test_strict_fixture_selection.py` 为 `6 passed`。
- 导入脚本 `TailGuardKV-labeling/scripts/import_labeled_fixtures_into_repo.sh` 已改为：对每个 fixture 先读取 manifest 的 `fixture_hash`、计算 SHA-256 并比对，再执行主仓库 `import_external_fixtures.py --validate-only`，全部通过后才导入；对应脚本测试为 `1 passed`。
- 隔离分支 `feat/external-baseline-flow` 的两个正式 smoke 配置已统一为：`CUDA_VISIBLE_DEVICES=0,1`、`profile_chunk_size: 1`、`use_persistent_workers: true`、`vllm_gpu_memory_utilization: 0.55`、`vllm_max_model_len: 1024`；配置测试为 `4 passed`。
- `TailGuardKV-labeling/configs/hybrid_session_candidates.yaml` 也已改为 `profile_chunk_size: 1` 和 `use_persistent_workers: true`，并已由 `validate_hybrid_measurement_config` 验证。

### 当前真实测量状态

- 本轮运行根目录：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/external_baseline_20260903/`。
- 已生成 1,950 条全量 LongBench QA/Summary 候选，并分为 33 个确定性、每批 60 请求的预筛批次：`prescreen/longbench/`。
- 已完成并严格合并的批次：`longbench_prescreen_batch000`、`longbench_prescreen_batch001`、`longbench_prescreen_batch002`。每批均生成 8-profile CSV 与 `.prescreen.json`，合并器未报告候选 hash、重复 request/profile、`ok/measured` 或覆盖错误。
- batch000--002 合并后的直接预筛计数为：`kivi_sensitive/qa=5`、`h2o_sensitive/qa=4`、`low_risk/qa=150`；所有 Summary cell 仍为 `0`。这些仅可用于候选排序，严禁写入最终 fixture 的 `risk_family`。
- `longbench_prescreen_batch003` 已启动，PID 为 `55079`；PID 文件：`prescreen/longbench/pids/longbench_prescreen_batch003.pid`，日志：`prescreen/longbench/logs/longbench_prescreen_batch003.log`，输出 CSV：`prescreen/longbench/longbench_prescreen_batch003.csv`。启动时使用 `flock` 锁 `prescreen/longbench/locks/longbench_prescreen_gpu.lock`，不得与任何其他 GPU profile/smoke 任务并发。

### 后续会话的严格执行顺序

1. 首先读取 batch003 的 PID；若进程仍存活，只等待它退出，不得启动新 GPU 任务。进程退出后，检查日志、CSV 是否存在和 GPU 是否释放。
2. 对已退出的批次，使用 `scripts/partition_labeling_batches.py` 的 `--launch-next` 路径写入对应 `.prescreen.json`，再不带 `--launch-next` 重算 `hybrid_need_status`。`--launch-next` 在存在 CSV 时只合并而不启动新的 GPU 批次；仍须核对输出中的 `launched: null`。
3. 若状态为 `pending`，仅启动 `next_batch_id` 对应的一个 60-request 预筛任务；命令必须继续使用 `nohup setsid flock -n ... env CUDA_VISIBLE_DEVICES=0,1 conda run -n tailguardkv-base python -m run_util.build_profile_table ...`，并保存 PID/日志。重复步骤 1--3，直到 `need_met` 或 `longbench_exhausted`。
4. 只有预筛满足 six-cell need 后，才调用 `select_hybrid_candidate_batches.py` 构造一个 12-session/60-request final-form hybrid batch。每个 hybrid batch 也必须完整实测 8 个 profile，再由 `label_sharegpt_sessions.py` 严格筛选；预筛风险绝不能继承为最终风险标签。
5. QA 只有在全部 LongBench QA 批次已完成、仍缺严格 QA hybrid cell，且 `raghot_qa_fallback_verified` 成功时，才允许使用 RAGhot QA。Summary 永远不得使用 RAGhot；来源耗尽仍不能满足任一 cell 时，写不足报告并终止，不能导出 fixture。
6. 仅当最终质量 fixture 同时满足 180 行（KIVI/H2O/low-risk 各 60，calibration/eval 各 90）且 session fixture 满足 48x5、12 个 `risk/task/split` cell 各 4 session、完整 8-profile 覆盖时，才能执行导入脚本。导入前保持 SHA-256/manifest 和 `--validate-only` 双重校验。
7. 正式 smoke 必须先 quality，且 `baseline_quality_signal_gate.passed=true` 后才允许 session。任一 gate 输出 `risk_evidence_insufficient` 或 `not_evaluated` 时，停止 policy 优劣比较，仅保留诊断。
8. 只有两个正式 smoke 均完成并通过各自 gate 后，才执行收尾：将最后两个有效结果和 baseline smoke 归档至 `out-备份/`，清理 `out/`，并把 `TailGuardKV-external-baseline`、`TailGuardKV-labeling` 中的关键代码、测试、脚本与配置迁入主仓库、提交并推送 GitHub。归档或迁移前不得删除任何隔离目录。

---

## 运行账本更新（2026-09-04，LongBench 预筛 supervisor）

- `TailGuardKV-labeling/scripts/partition_labeling_batches.py` 新增 `--run-until-terminal` supervisor。该入口应通过一次 `nohup setsid conda run -n tailguardkv-base ...` 异步启动；supervisor 在自身会话内串行前台运行单个 batch，进程返回后才读取对应 CSV、执行严格合并并重算 `hybrid_need_status`。
- supervisor 不跟随、不解析 profile 日志，也不通过轮询检测进度。每一轮仅接受已经形成的 CSV；缺 CSV、缺 profile、重复 request/profile、candidate hash 不一致或 `ok/measured` 不完整均会使严格合并失败并终止 supervisor，不会启动下一批。
- supervisor 的唯一正常终止条件为 `need_met` 或 `longbench_exhausted`。到达终态前，它会自动继续下一个 LongBench batch；因此不得再由会话手动以 `--launch-next` 衔接该运行根目录。
- 当前 supervisor：PID `28000`，PID 文件为 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/external_baseline_20260903/supervisor/longbench_prescreen_supervisor.pid`。它从已结束的 `longbench_prescreen_batch007.csv` 接续。控制输出保留在同目录，但在运行期间不读取 profile 日志。
- 聚焦验证：`conda run -n tailguardkv-base python -m pytest tests/test_labeling_batches.py -q`，结果 `13 passed`。覆盖 supervisor 在达到严格 gate 后停止、严格合并后的 prescreen manifest 写入，以及 batch 返回而未生成 CSV 时拒绝继续。

### Supervisor 续跑（2026-09-04）

- PID `28000` 已结束，但基于 manifest 的无日志状态重算仍为 `pending`，所以它未达到 `need_met` 或 `longbench_exhausted`。重算计数：`kivi_sensitive/qa=11`、`h2o_sensitive/qa=6`、`low_risk/qa=747`，三个 Summary cell 均为 `0`；下一批为 `longbench_prescreen_batch014`。
- 已确认 GPU 锁释放且没有残留 profile/supervisor 进程后，重新启动 supervisor，当前 PID `44743`，沿用相同 PID 文件与控制输出路径。它从 batch014 自动接续；运行期间继续禁止读取 profile 日志，也不得以手动 `--launch-next` 干预。

---

## 完成性审计（2026-09-04）

### 结论

- **全流程未完成，禁止导入正式 fixture、禁止启动正式 smoke、禁止得出 policy 比较结论。** GPU 空闲不是完成条件；必须满足 strict fixture、8-profile 覆盖与 smoke gate。

### 无日志证据

- supervisor PID `44743` 已退出，LongBench GPU 锁已释放；`nvidia-smi` 显示 3 张 GPU 均为 `0%` 利用率、`1 MiB` 显存占用，且未发现残留 profile/supervisor 进程。
- 从已写入 prescreen manifest 重算，状态仍为 `pending`，并非 `need_met` 或 `longbench_exhausted`：`kivi_sensitive/qa=11`、`h2o_sensitive/qa=6`、`low_risk/qa=747`，三个 Summary cell 均为 `0`；下一批应为 `longbench_prescreen_batch014`。
- `longbench_prescreen_batch014.csv` 已存在但只有 **247 行**。完整 60-request x 8-profile CSV 应为 481 行（含表头）；该批未产生 `.prescreen.json`，所以其不完整数据不得合并或用于风险计数。
- 现有 `baseline_quality_external.jsonl` 为 **84/180** 行：KIVI/H2O/low-risk 分别为 `12/12/60`，split 为 `42/42`，不满足严格 `60/60/60` 与 `90/90`。
- 现有 `baseline_session_external.jsonl` 为 **164/240** 行、**36/48** session，风险分布仅 `low_risk=164`，不满足 12 个 `risk/task/split` cell 各 4 session 的契约。
- 隔离主仓库 `out/` 未找到本计划所需的 `baseline_quality_signal_gate.json`、`risk_signal_gate.json` 或 `trace_semantics_gate.json`。此前目录中的旧 smoke 不能作为本方案的正式验收证据。

### 后续限制

1. 在定位并修复 batch014 未完成的根因前，不得重新启动 supervisor 或直接跳过 batch014；完整 CSV 是后续严格合并的前提。
2. 根因修复后，从 batch014 重新测量，只有形成完整 8-profile CSV 并写出严格 `.prescreen.json` 后，才能继续自动 supervisor。
3. 只有 six-cell prescreen need 满足后才构造 final-form hybrid batch；最终 fixture 必须先达成 180-row quality 与 48x5 session 契约，才允许执行导入校验和按 quality、再 session 的正式 smoke。

---

## 运行账本更新（2026-09-04，batch014 根因修复与重启）

### 根因与修复

- 已确认 `longbench_prescreen_batch014` 的 `full_gpu` 在 `longbench_qa_multifieldqa_en_000059` 发生 CUDA OOM；该记录为约 3,000 字符中文上下文，exact Qwen2 runtime 在 GPU 1 仅剩 1.04 GiB 时还需要分配 1.26 GiB。其后的七个 profile 因此没有测量行；supervisor 严格合并拒绝该不完整 CSV 是正确行为。
- 根因是 LongBench 候选构造器的 3,000 字符上限与 1,024-token 测量容量不匹配，且 JSONL fixture 路径不经过 ShareGPT 专用 token filter。另发现 `format_longbench_prompt` 只截断 context，超长 question 可以突破总 prompt 上限。
- 标注工作区已将 LongBench 默认 `max_prompt_chars` 收紧为 768，并在候选读取时拒绝总 prompt 仍超过上限的行；`configs/longbench_labeling.yaml` 已显式设置 `profile_chunk_size: 1` 和 `use_persistent_workers: true`。聚焦回归为 `16 passed`。

### 重启边界

- 保留 `artifacts/external_baseline_20260903/` 及 batch014 的 CSV、failed-chunks CSV、日志和 supervisor traceback，作为不可修改的失败证据；不再从该运行根续跑，也不混用其 candidate hash 或 prescreen manifest。
- 新运行根为 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/external_baseline_20260904/`。已重新生成并验证 1,902 条候选（QA 950、Summary 952、每条 prompt 最长 768 字符），并生成 32 个确定性 60-request prescreen batch；尚未生成任何新的 profile CSV 或 `.prescreen.json`。
- 新 supervisor 只能从 `longbench_prescreen_batch000` 开始串行运行；到达 `need_met` 或 `longbench_exhausted` 前不得手动 `--launch-next`、不得读取运行中的 profile 日志、不得并发启动其他 profile/smoke GPU 任务。
- supervisor 已通过一次 `nohup setsid conda run -n tailguardkv-base ... --run-until-terminal` 异步启动，PID 为 `35232`；PID 文件为 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/external_baseline_20260904/supervisor/longbench_prescreen_supervisor.pid`，控制输出为同目录 `longbench_prescreen_supervisor.log`。
- strict fixture、导入校验和正式 smoke 的所有既有限制继续有效；当前仍禁止导入 fixture、启动正式 smoke 或得出 policy 比较结论。

---

## 运行账本更新（2026-09-05，LongBench 预筛终态）

### 终态证据

- supervisor 已正常写出 `longbench_exhausted`，新运行根的 32 个 deterministic batch 均已生成严格 `.prescreen.json`；两张测量 GPU 已释放，未发现残留 profile 或 supervisor 进程。
- six-cell 直接预筛计数为：`h2o_sensitive/qa=19`、`h2o_sensitive/summary=26`、`kivi_sensitive/qa=7`、`kivi_sensitive/summary=10`、`low_risk/qa=841`、`low_risk/summary=869`。
- 因而严格 gate 未达到每 cell 12 条：缺 `kivi_sensitive/qa` 5 条、`kivi_sensitive/summary` 2 条。终态控制输出保存在 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/external_baseline_20260904/supervisor/longbench_prescreen_supervisor.log`。

### 严格处置

- 本轮 LongBench 来源已耗尽且 Summary 缺口存在；根据任务 4，Summary 禁止使用 RAGhot 或 tie 回填。因此本轮不能构造 final-form hybrid batch，必须以来源不足终止，不生成 session fixture。
- RAGhot QA 回退也未被触发：其前提是所有 LongBench QA batch 完成后，最终 hybrid 的严格 QA cell 仍不足；当前在 Summary 的直接预筛 gate 已终止，尚未产生任何 final-form hybrid 测量，不能把 direct prescreen 标签或不足直接转为 QA 回退依据。
- 不得导入 `baseline_quality_external.jsonl` 或 `baseline_session_external.jsonl`，不得启动 quality/session smoke，且不得得出 policy 比较结论。后续若要恢复流程，必须先由用户批准新的、可追溯的 Summary 来源或放宽已经写明的 strict coverage 目标；两者之外不得绕过本终态。

---

## 运行账本更新（2026-09-05，批准的混合预筛 reserve 放宽）

### 用户批准的范围

- 用户已明确批准仅放宽 direct prescreen reserve：`kivi_sensitive/qa` 从 12 至 7，`kivi_sensitive/summary` 从 12 至 10；H2O 与 low-risk 的四个 cell 仍为 12。
- 该变化只决定能否进入 final-form hybrid 测量。final risk 仍必须由每个 session 的 turn2--turn4、完整 8-profile 测量重新分类；48-session fixture 的 12 个 `risk/task/split` cell 各 4、唯一 provenance、tie 拒绝、导入校验与 smoke gate 均未放宽。

### 已执行

- `partition_labeling_batches.py` 已使用 cell-specific resolver 重算 32 个 strict prescreen manifest，输出 `need_met`：QA 7/7、Summary 10/10，其他四 cell 均至少 12；聚焦测试 `test_labeling_batches.py` 为 `15 passed`。
- selector 已修复为 LongBench-first：从 `.prescreen.json` 的确定性排序选取未使用 LongBench 内容，只有一个 batch 所需 LongBench QA 不足时才允许既有 RAGhot QA 回退；排序不会将 direct risk 标签带入最终候选。关联测试为 `41 passed`。
- 已构造独立 `hybrid_candidates_batch000`：12 个五轮 session、60 条记录、QA 6 / Summary 6、全部内容来源为 LongBench、12 个 original skeleton/content/payload 均唯一。batch 配置已验证 `profile_chunk_size: 1`、persistent worker 与双卡可见性。
- 已通过一次 `nohup setsid flock -n ... CUDA_VISIBLE_DEVICES=0,1 conda run -n tailguardkv-base python -m run_util.build_profile_table` 启动该 batch；PID 为 `64281`，PID 文件为 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/external_baseline_20260904/hybrid/pids/hybrid_candidates_batch000.pid`，日志为同目录 `logs/hybrid_candidates_batch000.log`。

### 运行限制

- 运行期间不得读取 profile 日志、不得启动第二个 GPU 任务；只在该 PID 结束后检查 CSV、failed-chunks 诊断和严格 8-profile 合并结果。
- 本 batch 结束且形成完整结果前，仍禁止导入 fixture、启动正式 quality/session smoke 或形成 policy 比较结论。
