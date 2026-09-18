# TTFT 坍缩修复实施计划（单 seed）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复固定 baseline 的动态行为、预算信号混淆、紧预算失效和 TTFT 图表尺度遮蔽，并以单个 seed 完成可审计的 session25 policy sweep。

**Architecture:** 保持现有 policy、backend、aggregation、visualization 分层。固定策略在构造期从该 seed 的 calibration profile 全表选择一次 profile；动态策略继续在运行时切换。预算字段由 policy projection 与 backend replay 两条独立链路产生，预检门禁通过后才启动正式网格。

**Tech Stack:** Python、pytest、CSV/JSON、Matplotlib；GPU 实验使用 `nohup setsid` 异步启动。

## Global Constraints

- 本轮只使用 `seed20260906`（1 seed），不运行 3-seed 扩展。
- policy 仅覆盖 `full_lru`、`static_best`、`static_safe`、`utility_dynamic`、`uncalibrated_dynamic`；不纳入 `tailguard`、`quality_oracle`。
- 正式矩阵为 `1 seed × 2 budgets × 2 epsilons × 2 deltas × 5 policies = 40 policy-cells`。
- 固定策略不得因请求、预测、缓存状态或预算切换 profile；允许 backend 自然 eviction/recompute。
- 产物写入 `out/full_25session_baseline_9_14/full/` 下新的 attempt/run 目录，不覆盖 retry2 原始结果。
- 诊断和实验输入、split、profile 表、版本信息必须可复现并写入 manifest。

---

### Task 1: 固定策略契约与单元测试

**Files:**
- Modify: `policies/base.py`, `policies/static_best.py`, `policies/static_safe.py`
- Test: `tests/test_policy_quality_audit.py`, `tests/test_session_aggregation.py`（仅新增与策略字段相关的测试）

- [ ] **Step 1: 写失败测试**
  - 构造含 exact 与多个 lossy profile 的 calibration 表。
  - 断言 `static_best` 构造后只选择一次满足平均质量约束且有 TTFT 收益的 profile。
  - 断言改变 request prediction、缓存状态或 budget 后，连续 `decide()` 的 profile 不变。
  - 断言 `static_safe` 只在 lossy 候选中应用 `mean_loss <= epsilon` 与 `violation_rate <= delta`，按最低平均 loss 选择；无合格候选时固定 `full_gpu`，且不产生 conformal fallback。
- [ ] **Step 2: 运行定向测试并确认失败**
  - Run: `pytest -q tests/test_policy_quality_audit.py -k 'static_best or static_safe'`
- [ ] **Step 3: 实现最小修复**
  - 将完整 calibration profile 表传入构造期选择函数。
  - 将选择结果存为不可变的 `selected_profile`/等价字段；`decide()` 只返回该 profile 的动作。
  - 明确记录 `selection_reason`、`exact_fallback`，固定策略不调用 conformal runtime fallback。
- [ ] **Step 4: 运行测试并确认通过**
  - Run: `pytest -q tests/test_policy_quality_audit.py -k 'static_best or static_safe'`

### Task 2: 拆分预算与运行时成本字段

**Files:**
- Modify: `policies/base.py`, `run_util/run_policies.py`, `metrics/collector.py`, `run_util/validation.py`
- Test: `tests/test_policy_quality_audit.py`, `tests/test_qwen2_session_runtime.py`

- [ ] **Step 1: 写失败测试**
  - 动作候选被 projected memory 拒绝时，仅断言 `policy_budget_filtered=True`。
  - backend replay 达到显存预算时，仅断言 `backend_budget_hit=True`。
  - 验证二者可同时为 false/true，旧的动作级 `budget_hit` 不得隐式串联两者。
  - 验证汇总分别统计 runtime profile switch、released KV、recompute、eviction。
- [ ] **Step 2: 运行定向测试并确认失败**
  - Run: `pytest -q tests/test_policy_quality_audit.py tests/test_qwen2_session_runtime.py -k 'budget or eviction or recompute or switch'`
- [ ] **Step 3: 实现字段传播与汇总**
  - 从 action decision 只写 `policy_budget_filtered`。
  - 从 backend replay 只写 `backend_budget_hit`，并保留逐事件计数。
  - 汇总 schema 增加四类成本字段及 runtime switch 次数/比例，保持旧字段兼容读取但不再推导新字段。
- [ ] **Step 4: 运行相关测试**
  - Run: `pytest -q tests/test_policy_quality_audit.py tests/test_qwen2_session_runtime.py`

### Task 3: 两档紧预算预检门禁

**Files:**
- Modify: `scripts/preflight_session25_fourth.py`, `configs/pilot_diagnostic_session25_fourth.yaml`
- Create: `tests/test_session25_fourth_preflight.py`（若现有测试不足）

- [ ] **Step 1: 写门禁测试**
  - 使用合成 profile/backend 记录验证两档 budget 均能让每个 lossy policy 触发 policy filtering 或 backend budget event。
  - 验证任意时刻 global resident 不超过预算。
  - 缺少触发、超预算或 schema 不完整时返回非零并写审计 JSON，且正式网格脚本不被调用。
- [ ] **Step 2: 实现可复现 budget 生成与审计**
  - 固定记录两档 budget 数值、seed、输入 profile 表版本和选择规则。
  - 输出每个 policy/budget 的 `policy_budget_filtered_count`、`backend_budget_hit_count`、`max_global_resident_mib`、`passed`。
- [ ] **Step 3: 运行门禁测试**
  - Run: `pytest -q tests/test_session25_fourth_preflight.py`

### Task 4: 单 seed sweep 配置与异步启动器

**Files:**
- Modify: `scripts/run_session25_fourth_sweeps.py`, `scripts/run_session25_fourth_full.sh`, `configs/baseline_wide_sweep.yaml`
- Test: `tests/test_session25_fourth_sweeps.py`

- [ ] **Step 1: 写矩阵测试**
  - 断言 seed 列表严格为 `[20260906]`，policy 数为 5，cell 数为 40。
  - 断言 `tailguard` 与 `quality_oracle` 被拒绝，输入/split/profile/version manifest 被保留。
- [ ] **Step 2: 实现配置与启动逻辑**
  - 正式网格启动前强制执行 Task 3 预检；预检失败直接退出，不创建正式 cell。
  - 新 attempt root 使用时间戳目录，例如 `out/full_25session_baseline_9_14/full/policy_attempts/single_seed_20260916/`。
  - GPU 命令采用：
    `nohup setsid bash scripts/run_session25_fourth_full.sh ... > <log>/grid.log 2>&1 & echo $! > <log>/grid.pid`
  - 同步记录 `<log>/grid.exit`，但运行期间不轮询，由用户在进程结束后通知。
- [ ] **Step 3: 运行配置测试**
  - Run: `pytest -q tests/test_session25_fourth_sweeps.py`

### Task 5: 聚合与双尺度图表

**Files:**
- Modify: `scripts/aggregate_baseline_wide_sweep.py`, `visual/plot_summary.py`
- Test: `tests/test_baseline_wide_sweep.py`

- [ ] **Step 1: 写图表与聚合测试**
  - 用含 200 ms 与 45 s TTFT 的合成记录，断言线性分面图和对数全景图都生成。
  - 断言两图包含全部 5 个策略、CI 和原始数值；对数图不丢弃离群值。
  - 断言聚合保留四类 runtime 成本、两类预算字段和单 seed provenance。
- [ ] **Step 2: 实现输出**
  - 保留 `summary_policy_p95_ttft.png` 作为局部线性分面主图。
  - 新增 `summary_policy_p95_ttft_log.png`，全量 TTFT 使用对数 y 轴并标注单位。
  - CSV/JSON/图表写入新的单 seed run 目录，保留原始数值和 CI 分母。
- [ ] **Step 3: 运行聚合测试**
  - Run: `pytest -q tests/test_baseline_wide_sweep.py`

### Task 6: 全量验证与结果验收

**Files:**
- Generated: `out/full_25session_baseline_9_14/full/<single-seed-run>/`
- Report: `out/full_25session_baseline_9_14/full/<single-seed-run>/root-cause-analysis.md`

- [ ] **Step 1: 运行完整相关测试**
  - `pytest -q tests/test_policy_quality_audit.py tests/test_qwen2_session_runtime.py tests/test_session25_fourth_preflight.py tests/test_session25_fourth_sweeps.py tests/test_baseline_wide_sweep.py`
  - `pytest -q`
  - `git diff --check`
- [ ] **Step 2: 在预检通过后异步启动 40-cell 单 seed 网格**
  - 只使用 `nohup setsid`，记录 PID、日志和版本信息；运行期间不轮询。
- [ ] **Step 3: 用户确认进程结束后聚合**
  - 验证 40 个 policy-cell、预算审计通过、resident 不超预算、失败状态未被当作成功样本。
  - 验证所有策略均出现于两张图，且 `static_best/static_safe` 无 runtime policy-induced profile switch。
- [ ] **Step 4: 编写根因报告并验收**
  - 报告区分 profile 固定选择、policy filtering、backend budget hit、释放 KV、recompute、eviction 与 TTFT 尺度问题。
  - 明确本轮为单 seed 诊断/验证结果，不宣称 3-seed 统计结论。

## 验收标准

- 固定策略构造后 profile 不随请求、预测、缓存状态或预算改变。
- `policy_budget_filtered` 与 `backend_budget_hit` 独立且均有逐事件来源。
- 两档紧预算预检失败时正式网格不会启动。
- 单 seed 正式矩阵恰为 40 个 policy-cells。
- 两张 TTFT 图均包含 5 个策略、CI、图例和原始数值。
- 新结果不覆盖 retry2 原始目录，所有输入与版本信息可复现。
