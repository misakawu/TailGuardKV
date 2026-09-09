# Session31 Batch7 双卡恢复实施计划
 
  ## Summary
 
  - 保留 Session31 已完成的 request/session ID 规范化与现有实验产物。
  - 删除 `batch007` 的三卡独立执行路径，使其与其他 batch 一样使用双卡。
  - Batch7 使用方案二的显存分配方式：
    - 显式构造双卡 `device_map`，不使用自动分片。
    - 模型层按现有双卡规则分配到逻辑 GPU 0、1。
    - 使用非对称 `max_memory`。现有单元测试期望值为 GPU 0 `8704MiB`、GPU 1 `9216MiB`。
  - Batch7 仍可单独补跑，但不再拥有不同的 GPU 拓扑或三卡专用配置。
 
  ## Implementation Changes
 
  ### 1. 固化方案二双卡运行约束
 
  在 `profiles/qwen2_runtime_common.py` 中复核并保持现有双卡加载行为：
- `CUDA_VISIBLE_DEVICES` 暴露两张物理卡后，运行时只使用逻辑设备 0、1。
  - `AutoModelForCausalLM.from_pretrained()` 接收显式 `device_map` 和 `max_memory`。
  - embedding、模型层、norm 和输出头均落在两张 GPU 上，不允许 CPU offload。
  - 奇数层模型保持确定性的非均匀切分，偶数层保持均匀切分。
  - 保留现有非对称显存预留策略，不为 Batch7 新增硬编码分支。
 
  公共接口不新增参数。Batch7 通过既有双卡加载接口运行，避免将实验批次概念下沉到模型运行时。
 
  ### 2. 删除 Batch7 三卡调度特例
 
  在 `scripts/run_diagnostic_session_batches.py` 中统一调度：
 
  - 删除或停用 `batch007` 对 `CUDA_VISIBLE_DEVICES=0,1,2` 的特殊处理。
  - 所有 batch，包括 `batch007`，生成相同形态的双卡子进程命令。
  - Batch7 可以继续作为单独恢复阶段运行，但环境、模型加载方式和显存策略必须与其他 batch 相同。
  - manifest 只记录批次、fixture、配置、输出目录和执行状态，不再声明三卡模式。
  - 恢复执行应采用新输出目录，禁止覆盖 `out/session31_full_rerun_20260908/` 中的旧现场。
  - 已完整通过产物校验的 batch 跳过，缺失或不完整的 `batch007` 才重新执行。
### 3. 更新执行计划与操作说明
 
  将原 `Session31-request-ID与Batch7三卡恢复计划-2026-09-08.md` 替换为双卡版本，至少更新：
 
  - 标题、Goal 和 Architecture 中的“三卡”描述。
  - 删除 GPU `0,1,2` 和 Batch7 三卡独占阶段。
  - 明确 Batch7 使用与其他 batch 相同的双卡拓扑，但采用方案二的显式映射和非对称 `max_memory`。
  - 写明恢复顺序：静态检查、单元测试、Batch7 小规模验证、Batch7 完整补跑、全局聚合校验。
  - 保留旧输出只读，使用带日期或恢复标识的新输出根目录。
 
  ## Test Plan
 
  ### 单元测试
 
  扩展 `tests/test_diagnostic_batch_runner.py`：
 
  - 生成包含 `batch007` 的 manifest。
  - 断言 Batch7 子命令不包含 `0,1,2`。
  - 断言 Batch7 与普通 batch 使用相同的双卡环境和命令结构。
  - 断言已有完整结果会跳过，缺失结果会重新调度。
续运行 `tests/test_qwen2_runtime_common.py`，确认：
 
  - 奇数层模型显式分布在逻辑 GPU 0、1。
  - 偶数层模型显式分布在逻辑 GPU 0、1。
  - `max_memory == {0: "8704MiB", 1: "9216MiB"}`。
  - `device_map` 中不存在第三张 GPU 或 CPU。
  - 单卡兼容路径不因本次调整发生回归。
 
  建议验证命令：
 
  ```bash
  pytest tests/test_qwen2_runtime_common.py \
         tests/test_diagnostic_batch_runner.py -v
  ```
 
  ### Batch7 运行验证
 
  1. 先使用 Batch7 fixture 执行最小规模或 profile-only 验证。
  2. 启动后检查进程仅可见两张 GPU。
  3. 检查两张 GPU 均加载模型分片，且没有第三张 GPU 参与。
4. 检查日志中无 CPU offload、设备不一致或 CUDA OOM。
  5. 验证通过后执行 Batch7 完整补跑。
  6. 对 Batch7 产物执行完整性校验，再与其他 batch 聚合。
  7. 聚合结果必须覆盖预期的全部 batch、session、request 和 profile，且不得混入失败运行的半成品。
 
  ## Acceptance Criteria
 
  - `batch007` 不再存在三卡代码路径、配置或运行说明。
  - Batch7 和其他 batch 都通过双卡子进程运行。
  - Batch7 模型加载采用显式双卡 `device_map` 和方案二非对称显存额度。
  - 相关单元测试全部通过。
  - Batch7 小规模实机验证无 OOM、CPU offload和设备映射错误。
  - Batch7 完整产物通过完整性检查并成功进入最终聚合。
  - 旧的 Session31 运行现场保持不变，可通过删除新恢复目录完成回滚。
 
  ## Assumptions and Unconfirmed Items
 
  - 默认继续使用现有测试锁定的 `8704MiB` 与 `9216MiB`，不在本次工作中重新调参。
  - “双卡”指子进程内的逻辑 GPU 0、1，具体对应哪两张物理 GPU 由启动环境决定。
  - 当前证据已确认运行时存在显式双卡映射和非对称显存测试，但尚未完成真实 GPU 上的 Batch7 小规模运行，因此实际峰值显存和是否彻底消除 OOM 仍需实机验证。
  - 当前处于计划阶段，尚未修改计划文档、代码或测试文件。


