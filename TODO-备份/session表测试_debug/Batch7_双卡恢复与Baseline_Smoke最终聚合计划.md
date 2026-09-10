# Batch7 双卡恢复与 Baseline Smoke 最终聚合计划

## Summary

- 保留现有 request/session ID 规范化结果以及已完成 batch 的实验产物，只补跑 `batch007`，不重跑其他成功批次。
- `batch007` 与其他 batch 统一使用双卡拓扑，不再维护三卡专用分支。
- 双卡加载采用显式 `device_map` 和非对称显存上限：逻辑 GPU 0 为 `8704MiB`，逻辑 GPU 1 为 `9216MiB`，禁止 CPU offload。
- 补跑成功后，将 Batch7 产物与其他 batch 合并，重新执行完整性、来源和诊断语义门禁，再生成最终 Baseline Smoke 汇总。
- 当前计划仅覆盖 Batch7 恢复及现有 diagnostic baseline 聚合，不把 `diagnostic_only` 结果升级为论文正式质量结论，也不在本任务中扩展新的 baseline 实现。

## Implementation Changes

### 1. 固化统一的双卡运行约束

- 在 `profiles/qwen2_runtime_common.py` 中保持确定性的双卡模型映射：
  - 运行时只使用逻辑设备 `0` 和 `1`。
  - embedding、模型层、norm 和输出头全部映射至两张 GPU。
  - `AutoModelForCausalLM.from_pretrained()` 显式接收 `device_map` 与 `max_memory`。
  - 不使用自动分片，不允许磁盘或 CPU offload。
  - 奇数层使用确定性的非均匀切分，偶数层均匀切分。
- 删除或禁用所有只针对 `batch007` 的三卡拓扑、第三张逻辑 GPU 映射和三卡显存覆盖�。
- 双卡映射必须为公共运行时行为，不能写成只对 Batch7 生效的特判。

### 2. 使用现有产物精确补跑 Batch7

- 以现有恢复目录为唯一输入来源：
  - `out/session32_request_id_batch7_recovery_20260908/manifest.json`
  - `out/session32_request_id_batch7_recovery_20260908/configs/batch007.yaml`
  - `out/session32_request_id_batch7_recovery_20260908/fixtures/batch007.jsonl`
- 通过现有 `only_batches={"batch007"}` 选择机制启动补跑，不重新物�化或重新编号 session。
- 保持 Batch7 中既有 request ID、session ID、输入顺序、模型、profile、policy、预算和输出路径语义不变。
- 启动前执行静态预检：
  - manifest 中恰好存在一个 `batch007`。
  - fixture 中 request ID 唯一且 session 归属未改变。
  - 配置引用的 fixture、profile table 和输出目录均指向 Batch7。
  - `skip_batches` 与 `only_batches` 不得同时包含 `batch007`。
- 执行结果写入新的补跑输出目录，�禁止覆盖现有 `out/` 产物。

### 3. 对 Batch7 执行分层验收

- 先运行最小真实 GPU smoke：
  - 模型能够在两张 GPU 上完成加载。
  - 实际设备映射中不存在逻辑 GPU 2、CPU 或 disk。
  - 至少完成一个请求的推理与结果落盘。
  - 无 CUDA OOM、设备不一致、缺失张量或跨卡索引错误。
- 最小 smoke 通过后，再执行完整 `batch007`。
- Batch7 完成条件：
  - 进程正常退出。
  - 预期请求全部产生结果，��不能静默漏行。
  - request ID 无重复、无缺失且与 fixture 一一对应。
  - session ID 没有被重写或漂移。
  - profile、policy、事件和来源字段齐全。
  - supervisor manifest 不得把运行时失败误标为纯门禁失败。
- 如果真实运行仍发生 OOM，停止最终聚合并保留失败证据。不得降低输入规模、改变 profile、丢弃请求或修改预算后把结果冒充原 Batch7。

### 4. 合并并重新生成 Baseline Smoke

- 仅在 Batch7 完整验收后，把新 Batch7 结果与原先成功 batch 合并。
- 聚合前按 `(request_id, profile, policy, experiment cell)` 检查：
  - 不存在重复记录。
  - 不存在缺失请求或缺失实验单元。
  - 各输入文件都有 `config`、`run_dir`、`diagnostic_only`、`quality_status` 和 `violation_status` 来源信息。
  - 同一聚合单元中的来源和诊断语义一致。
- 继续使用 `run_util/session_aggregation.py` 的现有聚合接口生成：
  - policy/cell 汇总 CSV。
  - session points CSV。
  - 事件 CSV。
  - Baseline Smoke Markdown。
  - 聚合 manifest 或 supervisor manifest。
- 最终 Markdown 保留现有核心列：
  - `memory_budget_mib`
  - `policy`
  - `p95_ttft_ms`
  - `mean_ttft_ms`
  - `mean_kv_cache_memory_mib`
  - `budget_hit_rate`
  - `restore_count`
  - `recompute_count`
  - `mean_quality_loss`
  - `quality_status`
- 所有本轮结果继续明确标记：
  - `diagnostic_only=true`
  - `quality_status=risk_evidence_insufficient`
  - `violation_status=risk_evidence_insufficient`
- 不得根据缺失或不足的风险证据计算正式 violation 结论，也不得移除 diagnostic 限定文字。

## Public Interfaces and Compatibility

- 保持现有 experiment runner、manifest 和 CSV 对外格式兼容。
- 保留 `select_manifest_batches(..., only_batches={"batch007"})` 与 `skip_batches` 的现有语义。
- 不改变既有 request/session ID 生成规则。
- 双卡 `device_map` 和 `max_memory` 应�由公共运行时方法统一生成；调用者只提供可见双卡环境，不新增 Batch7 专用公开参数。
- 聚合器继续拒绝缺少 provenance、诊断状态不一致或实验单元不完整的输入。

## Test Plan

### 单元测试

- 验证双卡映射只包含逻辑 GPU 0 和 1。
- 验证 `max_memory` 精确为 `{0: "8704MiB", 1: "9216MiB"}`。
- 验证 embedding、所有模型层、norm 和输出头均有显式设备归属。
- 验证偶数层和奇数层的切分均稳定��、可重复。
- 验证配置与加载参数中不存在 CPU、disk 或第三张 GPU。
- 保留并运行 Batch7 不漂移测试，确认 `batch007` 仍对应既有 session 集合。
- 验证 `only_batches={"batch007"}` 只选择 Batch7。
- 验证 `skip_batches` 与 `only_batches` 冲突时明确失败。
- 验证 Batch7 配置中的 profile table 和输出路径包含正确 batch 标识。
- 验证聚合器拒绝：
  - 缺少 provenance 的 CSV。
  - 重复 request/cell。
  - 缺失实验单元。
  - 混合 diagnostic 与非 diagnostic 数据。
  - 不一致的 `quality_status` 或 `violation_status`。

### 集成测试

- 使用临时 fixture 执行单 batch 流程，确认配置物化、运行、supervisor manifest 和聚合输出能够闭环。
- 运行 baseline 与 diagnostic runner 相关测试集，至少覆盖：
  - `tests/test_diagnostic_batch_runner.py`
  - `tests/test_baseline_wide_sweep.py`
  - `tests/test_external_baseline_configs.py`
- 完整测试通过后检查 `git status`，确认测试没有改写受版本控制的实验数据或用户文档。

### 真实 GPU 验收

- 双卡最小 smoke 通过后才运行完整 Batch7。
- 记录模型加载后的实际 device map、两卡峰值显存、进程退出码和失败日志。
- 完整 Batch7 通过后，检查请求计数、ID 集合及输出文件集合。
- 最终聚合后反向抽查至少一个 Batch7 request，确认其从 fixture、batch 输出、policy CSV 到最终汇总的来源链一致。

## Assumptions and Unconfirmed Items

- 默认复用当前 `out/session32_request_id_batch7_recovery_20260908` 中的 manifest、配置和 fixture，不重新构造 Batch7。
- 默认其他 batch 的成功产物不可修改，只在新的输出目录中补入 Batch7 并生成新汇总。
- 工作区目前包含多项既有删除和未跟踪内容，实施时不得恢复、覆盖、清理或提交这些与本任务无关的变更。
- 现有代码和测试已经具备 Batch7 单独选择、来源检查及 diagnostic 状态传播能力，本计划优先复用这些接口。
- 尚未得到真实 GPU 小规模运行证据，因此以下事项仍未确认：
  - `8704MiB + 9216MiB` 是否足以彻底消除 Batch7 OOM。
  - 实际加载后的 device map 是否完全符合设计。
  - Batch7 能否完整运行并进入最终聚合。
  - 最终 Baseline Smoke 是否能够通过全部完整性门禁。
- 上述未确认事项必须通过真实双卡 smoke 和完整 Batch7 补跑验证，不能仅凭单元测试宣称恢复成功。
