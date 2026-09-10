# Session31 Request ID 与 Batch7 三卡恢复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development`（推荐）或 `executing-plans` 逐任务实施。使用复选框跟踪步骤。

**Goal:** 修复 pressure trace 与原始 fixture 的 request/session ID 命名空间不一致，完成除 `batch007` 外的双卡 profile，并在 GPU `0,1,2` 上独立补跑 `batch007` 后回填完整诊断结果。

**Architecture:** 先停止当前混合版本实验并保留现场产物，在 pressure trace 边界建立共享 canonical ID 规则。新实验采用分阶段 supervisor：其他 batch 双卡 profile-only、`batch007` 显式三卡独立运行、最后校验已有产物并统一合并。

**Tech Stack:** Python、PyTorch、Transformers/Qwen2.5、YAML、pytest、现有 persistent worker 与 diagnostic batch supervisor。

## Global Constraints

- 当前 `session31_full_rerun_20260908` 进程在实施开始时直接停止，不等待自然结束。
- 保留 `out/session31_full_rerun_20260908/` 原样，禁止覆盖或删除。
- 新实验统一写入 `out/session32_request_id_batch7_recovery_20260908/`。
- 除 `batch007` 外保持 `balanced_two_gpu`；`batch007` 使用 `balanced_three_gpu` 和 GPU `0,1,2`。
- batch007 三卡结果回填 merged，但 merged 必须标记 `mixed_hardware=true`、`performance_comparability=diagnostic_only`。
- mixed-hardware 数据不得作为同硬件条件下的总体 TTFT、P95 或策略排序结论。
- 本轮不执行正式四档 B 性能验收，只进行 ID 修复验证、完整 profile 覆盖和 merged-input online smoke。
- 所有测试和实验均使用 `nohup setsid` 异步启动；运行期间不轮询，由用户通知进程结束。
- 未经用户明确授权，不执行 `git commit`。

---

## Interface Changes

- 在 `run_util/session_trace.py` 新增 `canonical_pressure_id(value: str) -> str`，统一移除尾部 pressure trace 后缀。
- `run_util.run_policies` 使用 canonical `(session_id, turn_index, request_id)` 匹配 evaluation measurements 与原始 fixture。
- Qwen runtime 新增正式配置值 `device_strategy: balanced_three_gpu`，严格管理三个可见逻辑 GPU。
- batch supervisor 新增：
  - `--skip-batches batch007`
  - `--only-batches batch007`
  - `--batch-overrides <yaml>`
  - `--validate-existing`
- batch 切分遇到最后一个 singleton session 时，将其合并进前一个 batch；27 个 session、目标每批 2 个时生成 12 个双-session batch 和 1 个三-session batch。

## Task 1：安全停止当前 Session31

**Files:**
- Read: `out/session31_full_rerun_20260908/full_rerun.pid`
- Preserve: `out/session31_full_rerun_20260908/`

- [ ] 从 PID 文件读取当前 supervisor PID。
- [ ] 验证 PID 对应命令包含 `session31_full_rerun_20260908`，避免终止无关进程。
- [ ] 向该 `setsid` 进程组发送 `SIGTERM`。
- [ ] 不轮询进程状态，由用户确认进程已经结束。
- [ ] 确认旧输出目录未被删除、移动或覆盖。

## Task 2：以测试驱动修复 ID 命名空间

**Files:**
- Modify: `run_util/session_trace.py`
- Modify: `run_util/run_policies.py`
- Modify: `scripts/run_diagnostic_session_batches.py`
- Test: `tests/test_qwen_session_backend.py`
- Test: `tests/test_diagnostic_batch_runner.py`

**Interfaces:**
- Produces: `canonical_pressure_id(value: str) -> str`
- Consumes: pressure measurement 中带后缀的 `session_id` 和 `request_id`
- Produces: 原始 fixture request，与 measurement evaluation membership 一一对应

- [ ] 增加失败测试：evaluation key 的 `session_id` 和 `request_id` 均带 `__pressure_r1_c1` 时，能够匹配无后缀 fixture。
- [ ] 断言返回的是原始 fixture request，其 prompt、reference、history 和无后缀 ID 不被 profile measurement 覆盖。
- [ ] 增加普通 ID 测试，确认不含 pressure 后缀时保持原值。
- [ ] 增加不同 round/copy 后缀测试，例如 `__pressure_r2_c3`。
- [ ] 增加真实缺失测试，确认 canonicalization 后仍不存在的请求继续抛出 missing-evaluation 错误。
- [ ] 增加 fixture canonical key 冲突测试；两个 fixture 请求规范化到同一 key 时，错误包含 `ambiguous canonical request keys`。
- [ ] 异步运行上述测试并确认修复前失败。
- [ ] 在 `run_util/session_trace.py` 实现共享 `canonical_pressure_id()`；只移除完整尾部 `__pressure_r<正整数>_c<正整数>`，普通 ID 中间的类似文本不得改变。
- [ ] 在 `_load_replay_inputs()` 构造 evaluation keys 时同时规范化 `session_id` 和 `request_id`。
- [ ] 在 `_load_online_evaluation_requests()` 构造 fixture keys 时使用相同规则，并在选择请求前检查 canonical key 歧义。
- [ ] 删除 supervisor 中重复的 pressure 正则和私有 canonical helper，改为导入共享函数。
- [ ] 异步运行目标测试，日志写入 `out/logs/session32_request_id_tests.log`。
- [ ] 验收测试全部通过，且现有无后缀 online fixture 测试保持通过。

## Task 3：补齐可恢复的 Batch Supervisor

**Files:**
- Modify: `scripts/run_diagnostic_session_batches.py`
- Test: `tests/test_diagnostic_batch_runner.py`

**Interfaces:**
- Produces: `--skip-batches`、`--only-batches`、`--batch-overrides`、`--validate-existing`
- Produces: 可分阶段执行并最终验证全部现有 batch 的 supervisor workflow

- [ ] 增加失败测试：27 个五轮 session、`sessions_per_batch=2` 时生成 13 个 batch。
- [ ] 断言 `batch000` 至 `batch011` 各有 2 个 session，最后的 `batch012` 有 3 个 session 和 15 个请求。
- [ ] 断言 `batch007` 仍包含原来的 session 14/15，不因尾批重平衡改变。
- [ ] 修改切分逻辑：只有末尾剩余一个 session 且存在前一批时，才将 singleton 合并进前一批；禁止复制、重叠或遗漏 session。
- [ ] 增加批次选择测试：`--skip-batches batch007` 不执行 batch007，但执行其他 batch。
- [ ] 增加批次选择测试：`--only-batches batch007` 只执行 batch007。
- [ ] `--skip-batches` 与 `--only-batches` 互斥；未知 batch ID 必须在启动实验前报错。
- [ ] 部分执行不得清理未选 batch 的已有输出。
- [ ] 部分执行只要所有选中 batch 成功即可返回 0，但不得生成 partial merged。
- [ ] 增加 batch override 测试：按 batch ID 对基础 YAML 做递归 mapping merge。
- [ ] override 应在写入 run-scoped fixture/output 之前应用；最终强制写回该 batch 的 `data.requests`、输出名和 `run_dir`，禁止 override 破坏 provenance。
- [ ] 增加 `--validate-existing` 测试：不调用 child runner，只验证 manifest 中全部 batch 的现有输出。
- [ ] 只有所有 batch coverage 完整时才能生成 merged；缺失 batch 时返回非零且不生成 partial merged。
- [ ] merged manifest 汇总每种 `(device_strategy, cuda_visible_devices)` 的 profile 行数。
- [ ] 检测到超过一种硬件条件时写入：

```json
{
  "mixed_hardware": true,
  "performance_comparability": "diagnostic_only"
}
```

- [ ] 异步运行 `tests/test_diagnostic_batch_runner.py` 并确认全部通过。

## Task 4：实现显式三 GPU Qwen 策略

**Files:**
- Modify: `profiles/qwen2_runtime_common.py`
- Test: `tests/test_qwen2_runtime_common.py`
- Create: `configs/pilot_diagnostic_session27_batch_overrides.yaml`

**Interfaces:**
- Consumes: `device_strategy: balanced_three_gpu`
- Produces: 三 GPU device map、max-memory map、peak/free/OOM diagnostics

- [ ] 增加失败测试：少于 3 个可见 GPU 时，`balanced_three_gpu` 给出明确错误。
- [ ] 增加未知 strategy 测试，禁止未知值静默按双卡执行。
- [ ] 增加 28 层三卡映射测试，层数按 `10/9/9` 分配：
  - GPU 0：layer 0–9，并放置 embedding。
  - GPU 1：layer 10–18。
  - GPU 2：layer 19–27，并放置 norm 和 lm_head。
- [ ] 使用 quotient/remainder 算法均匀分层，余数优先分配给编号更小的 GPU。
- [ ] 保持现有 `single_gpu` 和 `balanced_two_gpu` 的映射结果完全不变。
- [ ] 修改 managed-GPU 解析，使策略精确映射为 1、2 或 3 个逻辑 GPU，并验证可见设备数。
- [ ] `max_memory` 必须包含逻辑 GPU 0、1、2；GPU 2 使用默认 reserve `2048 MiB`。
- [ ] peak memory、free memory、OOM extra fields、peak reset、synchronize 和 CUDA cleanup 均遍历策略实际管理的 GPU。
- [ ] 三卡结果必须输出 `gpu2_peak_memory_mib`、`gpu2_used_mib`、`gpu2_free_mib` 和 `gpu2_total_mib`。
- [ ] 创建 batch override：

```yaml
batch007:
  profile_smoke:
    device_strategy: balanced_three_gpu
    cuda_visible_devices: "0,1,2"
```

- [ ] 异步运行 `tests/test_qwen2_runtime_common.py` 和 `tests/test_persistent_qwen2_worker.py`。
- [ ] 验收双卡行为无回归，三卡映射及诊断字段全部通过。

## Task 5：快速验证 Request ID 修复

**Inputs:**
- Config: `out/session31_full_rerun_20260908/configs/batch009.yaml`
- Measurements: `out/session31_full_rerun_20260908/batch_outputs/batch009/profile_tables/diagnostic_session27_profiles.csv`
- Output root: `out/session32_request_id_batch7_recovery_20260908/id_fix_smoke/`

- [ ] 使用旧 batch009 的完整 80 行 profile 表，异步启动一次 `run_util.run_policies`。
- [ ] 显式传入 `epsilon=0.05`、`delta=0.05` 和 `memory_budget_mib=4900`。
- [ ] 输出写入新 recovery 目录，禁止覆盖旧 batch009 结果。
- [ ] 不轮询，由用户通知 smoke 结束。
- [ ] 验收退出码为 0。
- [ ] 验收日志和输出中不再出现 `online Qwen fixture is missing evaluation requests`。
- [ ] 五个策略均产生 `online_qwen` measured records。
- [ ] 输出 request/session ID 使用原始无 pressure 后缀命名空间。
- [ ] 对 2-session 的 50/50 session split，预期 evaluation session 5 个 turn × 5 个策略，共 25 条 policy records。
- [ ] 若 smoke 失败，停止后续 profile 实验，返回 ID 数据流分析，不叠加 GPU 或 batching 修复。

## Task 6：运行除 Batch7 外的 Profile

**Output:** `out/session32_request_id_batch7_recovery_20260908/`

- [ ] 使用修复后的 supervisor 重新物化全部 13 个 batch。
- [ ] 使用基础配置 `configs/pilot_diagnostic_session27.yaml` 和 batch override 文件。
- [ ] 通过 `--profile-only --skip-batches batch007` 运行其他 12 个 batch。
- [ ] 其他 batch 沿用 `balanced_two_gpu`；batch007 override 此阶段只写入配置，不启动实验。
- [ ] 整个 supervisor 使用 `nohup setsid` 异步启动并保存 PID、stdout/stderr 日志。
- [ ] 不轮询，由用户通知运行结束。
- [ ] 验收 11 个普通双-session batch 各有 80 行 profile。
- [ ] 验收尾部三-session `batch012` 有 120 行 profile。
- [ ] 所有行必须满足 `ok=true`、`measured=true`。
- [ ] 不得存在 failed-chunks 记录或未知/重复 coverage。
- [ ] 此阶段不得生成 merged，因为 batch007 尚未完成。

## Task 7：GPU 0/1/2 独立补跑 Batch7

**Output:** `out/session32_request_id_batch7_recovery_20260908/batch_outputs/batch007/`

- [ ] 仅在其他 12 个 batch 全部通过后启动 batch007。
- [ ] 使用同一 recovery root、manifest 和 run-scoped batch007 config。
- [ ] 通过 `--profile-only --only-batches batch007` 独立运行。
- [ ] batch override 必须使 runtime 使用 `balanced_three_gpu` 和 `CUDA_VISIBLE_DEVICES=0,1,2`。
- [ ] 使用 `nohup setsid` 异步启动，单独保存 `batch007_three_gpu.pid` 和日志。
- [ ] 不轮询，由用户通知运行结束。
- [ ] 验收 profile 行数为 80。
- [ ] 验收 `hybrid-session-015-turn-4` 成功完成。
- [ ] 验收无 `profiles_failed_chunks.csv` 中的失败记录。
- [ ] runtime/worker provenance 必须明确记录三卡 strategy 和可见设备。
- [ ] GPU diagnostics 必须包含 GPU 0、1、2。
- [ ] 不得出现 CUDA OOM、fatal CUDA error 或 `worker_state_lost=true`。
- [ ] 若三卡仍 OOM，立即停止本计划后续合并；保存该请求三卡显存快照并重新进入 `systematic-debugging`，不得通过减少 token 数或修改数据内容掩盖问题。

## Task 8：验证并回填 Mixed-Hardware Merge

- [ ] 使用 `--validate-existing` 对 recovery root 的全部 13 个 batch 做只读验证。
- [ ] 验证阶段不得重新执行任何 GPU profile。
- [ ] 只有全部 batch 通过时生成 `merged/profile_tables/diagnostic_session27_profiles.csv`。
- [ ] merged profile 必须包含 27 个 session、135 个 request、8 个 profile，共 1080 行。
- [ ] 每个 canonical `(request_id, profile)` 必须恰好出现一行。
- [ ] 不得存在重复、缺失、失败、dry-run 或残留 failed-chunk 记录。
- [ ] batch007 行必须记录三卡 provenance，其余行记录双卡 provenance。
- [ ] merged manifest 必须写入 `mixed_hardware=true` 和 `performance_comparability=diagnostic_only`。
- [ ] 使用 merged input 运行一个单预算 online policy smoke，验证完整输入可消费且 ID 映射正确。
- [ ] 本轮禁止将 mixed-hardware 总体 TTFT、P95 或策略排序写成正式性能结论。
- [ ] 本轮不运行正式四档 B × 五策略性能 sweep；统一硬件重跑另立任务。

## Task 9：文档与最终验证

**Files:**
- Modify: `交接文档_2026-09-06_session27_online_baseline.md`

- [ ] 记录 pressure ID 根因：measurement evaluation keys 带 pressure 后缀，而 online fixture 使用原始 ID，旧 loader 进行严格未规范化匹配。
- [ ] 记录 session31 因继续运行会混用代码版本而被主动终止。
- [ ] 记录 27-session 尾部 singleton 的重平衡规则。
- [ ] 记录 batch007 双卡 OOM 与三卡复跑结果，包括三张 GPU 的峰值和空闲显存。
- [ ] 记录 mixed-hardware profile 仅具有诊断和覆盖意义。
- [ ] 异步运行相关回归测试：

```text
tests/test_qwen_session_backend.py
tests/test_diagnostic_batch_runner.py
tests/test_qwen2_runtime_common.py
tests/test_persistent_qwen2_worker.py
tests/test_qwen2_session_runtime.py
```

- [ ] 相关测试通过后，再按仓库既有命令异步运行全量测试。
- [ ] 不修复与本任务无关的测试失败；单独记录已知例外。
- [ ] 检查 `git diff`，确认未修改旧 session31 输出，未包含用户已有无关改动。
- [ ] 未经用户明确授权，不提交 commit。

## Final Acceptance Criteria

- ID 单元测试及 batch009 online policy smoke 证明 pressure measurements 能正确匹配原始 fixture。
- 27 个 session 全部进入新的13个 batch，且不存在单-session pressure batch。
- 除 batch007 外的 profile 在双卡条件下全部完整成功。
- batch007 在 GPU `0,1,2` 上产生完整80行 profile，原 OOM 请求成功。
- merged profile 恰好1080行，无重复、缺失或失败记录。
- mixed-hardware provenance 完整，所有汇总明确标记为 diagnostic-only。
- 旧 session31 失败现场完整保留。

## 当前执行进度（2026-09-08）

> 更新时点：2026-09-08 21:13（UTC+8）。以下状态仅依据恢复工作树、现有输出文件和测试日志记录。

### 总体状态

恢复工作目前处于“代码修复和单批 smoke 已验证，正式 profile 恢复尚未启动”的阶段。当前没有运行中的 Session32、diagnostic batch runner 或 `batch007` 恢复进程。

### 已完成

- Session31 已停止，旧输出和实验失败现场继续保留，未发现恢复任务覆盖旧 Session31 输出。
- 已在独立恢复工作树 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-session32-recovery` 中开展修复，未把本轮改动合并回主工作树。
- pressure evaluation key 与原始 fixture 的 request/session ID 命名空间匹配逻辑已经修复，并补充了对应测试。
- 27 个 session 的 batch 划分和尾部 singleton 重平衡逻辑已经实现并覆盖测试。
- diagnostic batch runner 已增加单 batch、排除 batch、batch 级 GPU override、输出命名和恢复运行所需能力，并补充相应测试。
- 已修复 `arrival_index` 在 Session runtime 路径中的传递问题，并完成相关回归验证。
- 已修复 KIVI mask contract 问题；KIVI 定向测试及关联回归测试通过。
- 相关回归测试最新结果为 `87 passed in 6.14s`，日志位于：
  `out/session32_request_id_batch7_recovery_20260908/kivi_mask_contract_tdd/related_regression_20260908_204502.log`。
- `batch009` online policy smoke 已在位置修复和 KIVI mask 修复后成功完成。最终日志与记录分别位于：
  - `out/session32_request_id_batch7_recovery_20260908/id_fix_smoke/batch009_policy_smoke_after_kivi_mask_fix_retry_20260908_205528.log`
  - `out/session32_request_id_batch7_recovery_20260908/id_fix_smoke/batch009_policy_records_after_kivi_mask_fix_retry_20260908_205528.csv`
- smoke 结果已产生有效 backend 语义证据，包括 session reuse、global resident evolution 和 backend event evidence；pressure ID 修复已不再被 fixture 命名空间不一致阻塞。

### 本轮新增修复

- `profiles/qwen2_kv_runtime.py`：补齐 Session runtime 所需的 position/arrival 信息传递。
- `profiles/kivi_cache.py`：修复 KIVI mask contract。
- `scripts/run_diagnostic_session_batches.py`：扩展 batch 恢复、GPU override、输出隔离和恢复执行能力。
- `tests/test_qwen_session_backend.py`、`tests/test_qwen2_session_runtime.py`、`tests/test_diagnostic_batch_runner.py`、`tests/test_kivi_cache.py`：增加或调整回归覆盖。

### 尚未完成

- 尚未正式启动除 `batch007` 外的双卡 profile 恢复运行。
- 尚未在 GPU `0,1,2` 上独立补跑 `batch007`，因此还没有验证原 OOM 请求是否成功，也没有形成完整80行 `batch007` profile。
- 尚未采集并回填三张 GPU 的峰值显存及运行前后空闲显存。
- mixed-hardware provenance 和 diagnostic-only manifest 标记尚未通过最终产物验证。
- 尚未生成并核对包含27个 session、135个 request、8个 profile、共1080行的 merged profile。
- 尚未验证 merged profile 是否无重复、无缺失、无失败记录。
- 完整相关测试已通过，但仓库全量测试尚未执行。
- 当前代码、测试和本文档改动仍为本地未提交状态；本轮未提交、未推送，也未合并回主工作树。

### 下一步执行顺序

1. 在双卡环境运行除 `batch007` 外的正式 profile，并逐 batch 核对完成状态与行数。
2. 清理运行残留进程和显存后，在 GPU `0,1,2` 上单独执行 `batch007`。
3. 记录三卡显存证据，确认原 OOM 请求成功，并核对 `batch007` 恰好80行。
4. 合并全部 profile，核对27个 session、135个 request、8个 profile和1080行总量。
5. 验证 mixed-hardware provenance、diagnostic-only 标记、重复项、缺失项和失败项。
6. 运行仓库全量测试，检查最终 `git diff`；未经用户明确授权不提交 commit。
