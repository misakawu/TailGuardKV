# baseline smoke 修复计划

## Summary

目标是把当前 `out/20260808_193926_pilot` 暴露出的三类问题一次性修平：

1. exact `full_gpu` profile 丢失会话内驻留语义；
2. baseline smoke 里的 `static_best/static_safe/uncalibrated_dynamic` 在当前数据形状下全部退化到 `full_lru`，导致表失去区分度；
3. `budget_hit` 与事件统计口径混杂，summary 曲线不再能直接解释 backend 行为。

修复完成后，baseline smoke 表需要能稳定回答四件事：会话内驻留、跨会话全局驻留、预算命中、`restore/recompute/evict` 行为；并且 5 个 baseline 在 smoke 表上应有真实可解释的分化，而不是大面积重合。

## Implementation Changes

### 1. 接通 exact profile 的 session runtime 链路

- 在 `profiles/base.py` 的 `transformers_profile_many_measurements()` 增加 `session_runtime` 和 `memory_budget_mib` 参数，行为与 `qwen2_kv_profile_many_measurements()` 对齐。
- 将 `session_runtime_state` 注入 `profiles.transformers_runtime` worker payload，并在 worker 返回时回收更新后的状态。
- 修改 `profiles/transformers_runtime.py`：
  - `run_profile_batch()` 读取输入中的 `session_runtime_state`；
  - 请求循环改为走 `_run_one_request_with_session()`，而不是直接 `_run_one_request()`；
  - 输出中补回 `session_runtime_state`，确保 chunk 间会话状态连续。
- 保证 exact profile 也稳定写出：
  - `kv_incremental_mib`
  - `kv_cumulative_mib`
  - `resident_kv_mib_before`
  - `resident_kv_mib_after`
  - `restore_ms`
  - `recompute_ms`
  - `evicted_kv_mib`
  - `budget_hit`

### 2. 统一 exact / lossy 的 measured profile 语义

- 对齐 `transformers_runtime` 与 `qwen2_kv_runtime` 的 session 语义：
  - turn 0 的 `resident_before=0`；
  - 同 session 后续 turn 继承前一 turn resident；
  - profile 切换时用统一的 restore/recompute 规则，不允许 exact 路径“无状态”。
- 明确 exact profile 的 `kv_cache_memory_mib` 与 `resident_kv_mib_after` 关系，避免 summary 层把一个当瞬时峰值、另一个当驻留值。
- 保留现有 CSV 列名，不改外部消费接口，只修内部语义和实际填充值。

### 3. 修正 baseline 退化的设计问题，而不是只补表面分支

- 保留 `static_best/static_safe/uncalibrated_dynamic` 的 baseline 定义，但把 smoke 阶段的选择逻辑改为基于可部署 profile 子集和真实 calibration 形状做稳定分化。
- `StatsPolicy` 需要区分两件事：
  - “tail-SLO 下安全”
  - “smoke baseline 需要保留可比较行动空间”
- 具体调整：
  - `static_best` 继续表示“平均质量约束下最快固定 profile”，但不能因为当前 `epsilon` 极小就无条件退回 exact；应优先在 lossy 子集中找满足 baseline 约束的固定 profile，只有无可用 profile 时才回退 exact。
  - `static_safe` 继续表示“tail-SLO 下最保守固定 profile”，允许回退 exact，但要把“无 safe lossy profile”显式记录到 reason / fallback 字段。
  - `uncalibrated_dynamic` 保持“点预测阈值、不做 conformal”的定义，但不能因为 exact 总是 safe 就在 smoke 表里完全吞掉 lossy 路径；应先在 lossy 候选中做点预测筛选，只有无候选才 fallback exact，并把 fallback 原因稳定写出。
- 不改 `utility_dynamic` 的基本语义：它仍然是不受校准约束的平均 utility baseline，用来提供“收益高但风险差”的对照面。
- 如果当前 pilot 数据本身确实不存在任何满足 `epsilon=0.05/0.1` 的 lossy profile，则 smoke summary 中必须显式暴露这一事实，而不是只让三条 baseline 静默变成 `full_lru`。

### 4. 拆分 policy 预算过滤与 backend 预算事件

- 在 `PolicyRunRecord` / `ActionDecision` / summary 链路里拆开两个字段：
  - `policy_budget_filtered`：策略侧是否因 projected memory 过滤过候选；
  - `backend_budget_hit`：backend replay 是否真的触发预算压力。
- `budget_hit_rate` 在 summary 中只统计 backend 事件，不再混入策略过滤信号。
- 如需保留策略视角统计，新增独立 summary 字段，例如 `policy_budget_filter_rate`。
- `run_util/core_types.py` 中 `backend_result.budget_hit or action.budget_hit` 的合并逻辑改为分别保留，避免语义污染。

### 5. 对齐 projected memory 与 replay 状态语义

- 复查 `policies/full_lru.py` 和 `StatsPolicy._projected_memory()`，统一“projected resident”的定义，避免策略预测显著偏离 backend 回放。
- `full_lru` 的 projected memory 计算改成和 measured replay 同一口径：基于当前 session resident、增量 KV 和全局 resident 投影，而不是重复累加 `current_cumulative`。
- exact / lossy policy 的 projected memory 都以 replay-visible state 为准，确保 `budget_hit`、fallback 和 action selection 用的是同一套内存语义。

## Public Interfaces / Types

- `transformers_profile_many_measurements(..., session_runtime=None, memory_budget_mib=None)`：新增与 lossy adapter 对齐的会话参数。
- `profiles.transformers_runtime.run_profile_batch()`：输入/输出新增 `session_runtime_state`。
- `ActionDecision` / `PolicyRunRecord` / policy CSV：
  - 新增 `policy_budget_filtered`
  - 新增 `backend_budget_hit`
  - `budget_hit` 若保留，定义为 backend 事件别名，不能再表示混合语义。
- summary CSV：
  - `budget_hit_rate` 重新定义为 backend 预算命中率；
  - 新增 `policy_budget_filter_rate`；
  - fallback reason / safe candidate 信息保留，便于解释 baseline 退化。

## Test Plan

- `transformers` session runtime 单测：
  - 同 session 第二 turn 的 `resident_kv_mib_before == 前一 turn 的 resident_kv_mib_after`
  - profile 切换时 exact 路径产生 restore 或 recompute 事件
  - chunk 边界前后 session state 连续
- replay/backend 单测：
  - exact profile 表不再出现 `resident_*` 为空而 lossy 非空的分裂
  - 新 session `resident_before=0`，同 session后续 turn 可继承 resident
  - `global_resident_kv_mib` 可跨 session 维持非零
- policy 单测：
  - `static_best/static_safe/uncalibrated_dynamic` 在 smoke 数据上不再静默退化成与 `full_lru` 完全同分布，除非数据确实无可选 lossy profile；若无，则 reason/fallback 字段必须明确说明
  - `utility_dynamic` 继续保持 lossy 选择能力
  - `full_lru` projected memory 与 backend 投影一致
- summary 单测：
  - `budget_hit_rate` 只反映 backend 事件
  - `policy_budget_filter_rate` 与 `budget_hit_rate` 可独立变化
  - `restore/recompute/evict` 统计与逐条 policy 表一致
- 端到端验证：
  - 重新运行 `pilot-smoke-measured`
  - 检查 5 个 baseline、2 个 budget、4 个 `epsilon/delta` 组合全部产出
  - 核查 exact profile 在 profile 表和 policy 表中的 session resident 字段不再缺失
  - smoke 表能直接解释四类行为：会话内驻留、跨会话全局驻留、预算命中、`restore/recompute/evict`

## Assumptions

- 不改变现有主入口命令、CSV 主文件名和已有列的基本兼容性；新增字段只做追加。
- `utility_dynamic` 仍然是“不保 tail-SLO 的平均收益 baseline”，不强行改成安全策略。
- 如果 pilot 数据在当前 `epsilon=0.05/0.1` 下确实没有任何 safe lossy profile，允许 `static_safe` 回退 exact，但必须在输出中把原因显式化。
- 本轮修复以 measured-replay smoke 语义正确为第一优先级，不扩展新的 profile 或算法。
