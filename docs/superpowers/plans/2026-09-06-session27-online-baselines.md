# Session27 Online Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` task-by-task. Steps use checkbox syntax.

**Goal:** 用真实 Qwen 持久会话 backend 重跑 session27 的五条 baseline，产出可解释的诊断曲线和事件链。

**Architecture:** 保留 profile 表作为 calibration 输入，新增在线 policy backend 直接执行 fixture 请求。策略每轮选择 profile 后由同一持久 worker 执行；profile 不兼容切换强制完整重算，LRU 与预算事件由 backend 真实维护。

**Tech Stack:** Python、PyTorch/Qwen2.5、现有 persistent worker、pytest、matplotlib。

## Global Constraints

- session27 所有输出必须标记 `diagnostic_only=true`。
- 所有质量和 violation 结果标记 `risk_evidence_insufficient`，不宣称 tail SLO。
- 固定实验种子 `20260906`；B 取无驱逐 full 扫描的 P25/P50/P75/P90。
- 禁止以 batch P95 平均值作为总体 P95。

### Task 1: 固定分层切分与预算扫描

**Files:** 修改 `run_util/experiment_common.py`、`configs/pilot_diagnostic_session27.yaml`；创建 `run_util/derive_session_budgets.py`；测试 `tests/test_policy_session_budget.py`。

- [ ] 先写测试：同一 `session_id` 不跨 calibration/evaluation；任务、长度桶、turn-depth 在两集合保持分层；相同种子结果稳定。
- [ ] 实现 `split_measurements(..., split_seed=20260906, stratify_session=True)`，按 session 分配 50/50。
- [ ] 实现 full 无驱逐扫描，输出 occupancy 序列和 P25/P50/P75/P90 MiB 到 JSON。
- [ ] 配置引用生成的四档 B，不再使用写死的 `[26, 40]`。
- [ ] 运行目标测试并提交 `fix: stratify session27 calibration and budgets`。

### Task 2: 真实在线 Qwen Session Backend

**Files:** 创建 `backends/qwen_session.py`；修改 `backends/base.py`、`run_util/run_policies.py`、`profiles/qwen2_kv_runtime.py`；测试 `tests/test_qwen_session_backend.py`。

- [ ] 先写测试：同 profile 连续 turn 复用 KV；不兼容 profile 切换产生 `recompute_ms`；global LRU 超 B 时产生 `budget_hit` 和 `evicted_kv_mib`。
- [ ] 定义 `OnlineQwenSessionBackend.execute(request: Request, action: Action, cache_state: CacheState) -> BackendResult`。
- [ ] 复用 persistent worker 和 fixture prompt/history；禁止从 `MeasuredReplayBackend` 返回 profile 表的 TTFT。
- [ ] 实现全局 LRU resident 索引；完整记录 before/after resident、global resident、evict、restore、recompute、queue。
- [ ] 在 `run_policies.py` 增加 `online_qwen` backend 配置，在线执行 evaluation fixture；replay 仅保留旧诊断兼容入口。
- [ ] 运行 backend 测试与一个 2-session GPU smoke，提交 `feat: run policies on persistent qwen backend`。

### Task 3: 五条策略语义修正

**Files:** 修改 `policies/base.py`、`static_best.py`、`static_safe.py`、`utility_dynamic.py`、`uncalibrated_dynamic.py`、`policies/registry.py`；测试对应 policy 测试文件。

- [ ] 为 `static_best` 写失败测试：lossy 未比 full 快 5% 时选 full；满足门槛时选最快 lossy。
- [ ] 允许 exact 进入 static_best 比较，采用校准集合并 P95 TTFT 点估计和 5% benefit gate。
- [ ] 为 `static_safe` 写失败测试：primary 是最小经验 quality-loss 的 lossy；conformal 不安全时最终 action 为 full，并保留 primary/fallback 字段。
- [ ] 为 uncalibrated 写失败测试：在 `pred_loss <= epsilon` 的 lossy 中按预测 TTFT 最小值选择，而不是按 loss 最小值。
- [ ] 将 utility_dynamic 改为全局 primal-dual：`ttft_norm + 0.25*kv_norm + lambda*(loss-epsilon)`，`lambda=max(0, lambda + 0.5/sqrt(t)*residual)`。
- [ ] 提交 `fix: align baseline policy semantics with experiment design`。

### Task 4: Shadow Audit 与逐轮质量记账

**Files:** 修改 `run_util/core_types.py`、`run_util/run_policies.py`、`run_util/experiment_summary.py`；测试 `tests/test_policy_quality_audit.py`。

- [ ] 写失败测试：固定种子下 10% audit 集合稳定；审计轮使用实际 loss 残差，非审计轮使用预测；inverse-probability 估计正确。
- [ ] 在 `PolicyRunRecord` 增加 `audit_selected`、`predicted_quality_loss`、`observed_quality_loss`、`quality_estimate`、`primary_profile`。
- [ ] 对 utility_dynamic 用校正质量更新 lambda；主表同时输出估计质量和 audit 样本数。
- [ ] 对 static_safe 的 full fallback 将最终执行 TTFT、KV、质量完整并入总计。
- [ ] 提交 `feat: record audited quality and final policy actions`。

### Task 5: 正确汇总与四张图

**Files:** 修改 `run_util/experiment_summary.py`、`visual/plot_summary.py`；测试 `tests/test_session_summary_visuals.py`。

- [ ] 写失败测试：构造不同 batch 大小的记录，验证总体 P95 不等于 batch P95 均值。
- [ ] 从 merged per-turn records 直接计算每个 `(B, epsilon, delta, policy)` 的 P95、均值、动作分布和 backend 事件。
- [ ] 采用 session block bootstrap 输出 95% 误差条；static_best 的选择仍使用点估计。
- [ ] 保持四张 `summary_policy_*.png`：主 policy 线加淡色 batch/session 散点；quality/violation 图显式显示 `risk_evidence_insufficient`。
- [ ] 增加事件汇总 CSV，联读 TTFT、budget hit、restore、recompute、queue。
- [ ] 提交 `fix: aggregate policy metrics from turn records`。

### Task 6: 端到端诊断运行与验收

**Files:** 修改 `scripts/run_diagnostic_session_batches.py`、配置和交接文档。

- [ ] 先运行 2-session、单 B、五策略 online smoke；验证无 replay source、每轮有 backend event 字段。
- [ ] 再异步运行 session27 四档 B；只在每批完整成功后合并结果。
- [ ] 验收：B 变化能在至少一个 backend 事件指标上体现；策略切换能看到恢复或重算；所有输出保留 diagnostic/risk provenance。
- [ ] 若 `full_lru` 仍最快或 B 对 TTFT 不敏感，输出事件分解作为结论，不调整策略制造预期排序。
- [ ] 提交 `test: validate online session27 baseline diagnostics`。
