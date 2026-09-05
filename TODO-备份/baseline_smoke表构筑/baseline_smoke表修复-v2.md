# 5 个 Baseline Smoke 表获取与验证计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 2026-08-15 截止日前，完成 5 个强制 Baseline（`full_lru`, `static_best`, `static_safe`, `utility_dynamic`, `uncalibrated_dynamic`）的 Smoke 表生成，确认所有策略在统一 Runner 下公平执行，并采集到用于 H0/H1/H2-lite 诊断的基础指标。

**Architecture:** 复用现有 `policies/registry.py` 工厂和 `run_experiment.py` 入口。通过配置文件显式限定策略集合（剔除 `tailguard` 与 `quality_oracle`），使用 `measured-replay` Backend 回放预计算的 Profile 表，确保 5 个策略共享完全相同的请求序列与 Profile 物理成本。

**Tech Stack:** Python 3.10+, PyTorch, vLLM/LMCache (抽象层), pandas (可选), yaml, pytest。

## Global Constraints

- 策略集合固定为以下 5 个：`full_lru`, `static_best`, `static_safe`, `utility_dynamic`, `uncalibrated_dynamic`。
- Profile 层必须包含 `full_gpu`, `kivi_4bit_residual32`, `kivi_4bit_residual64`, `h2o_heavy10_recent10`（至少 2 种有损 Profile 供静态/动态策略选择）。
- 使用 `configs/pilot.yaml` 中的现有 profile 配置，不新增 profile。
- 输出路径：
  - `out/profile_tables/pilot_smoke_measured_profiles.csv`
  - `out/policy_tables/pilot_smoke_measured_policy.csv`
  - `out/policy_tables/pilot_smoke_measured_summary.csv`
- 验证标准：每个策略必须生成不低于 50 条有效请求记录（`ok=True`），且没有任何策略因缺少测量数据而崩溃。

---

### Task 1: 配置隔离 —— 显式限定仅 5 个 Baseline

**Files:**
- Modify: `configs/pilot.yaml`
- Test: `tests/config/test_pilot_loading.py`（若存在，否则手动校验）

**Interfaces:**
- Consumes: YAML 解析逻辑 (`config_loader.py`)
- Produces: 策略列表仅含 5 个指定名称

- [ ] **Step 1: 编辑 `configs/pilot.yaml`**  
  找到 `policies` 配置段。将 `names` 或 `items` 修改为仅包含这 5 个策略：
  ```yaml
  policies:
    names:
      - full_lru
      - static_best
      - static_safe
      - utility_dynamic
      - uncalibrated_dynamic
  ```
  如果原来有 `tailguard` 或 `quality_oracle`，直接删除或注释掉。

- [ ] **Step 2: 验证 YAML 语法与加载逻辑**  
  执行 Python 命令：
  ```bash
  python -c "from experiment_common import config_policies, load_config; print(config_policies(load_config('configs/pilot.yaml')))"
  ```
  预期输出应为 `['full_lru', 'static_best', 'static_safe', 'utility_dynamic', 'uncalibrated_dynamic']`。

- [ ] **Step 3: 提交配置变更**  
  ```bash
  git add configs/pilot.yaml
  git commit -m "config: restrict to 5 baselines for smoke phase"
  ```

---

### Task 2: 重新生成 Profile 测量表（确保依赖数据完整）

**Files:**
- No code changes; this is an execution step.
- Consumes: `configs/pilot.yaml`, `profiles/` 层所有 Adapter
- Produces: `out/profile_tables/pilot_smoke_measured_profiles.csv`

- [ ] **Step 1: 清理旧的 Profile 表（避免混淆）**  
  ```bash
  rm -f out/profile_tables/pilot_smoke_measured_profiles.csv
  ```

- [ ] **Step 2: 运行 Profile 构建命令**  
  因为配置中的策略变了，但 Profile 集合没变，只需重新生成确保数据一致：
  ```bash
  python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
  ```
  观察输出日志，确认 `build_profile_table` 阶段完成，且所有 Profile（`full_gpu`, `kivi_*`, `h2o_*`）的 `measured` 列均为 `True`。

- [ ] **Step 3: 快速校验 Profile 表行数**  
  使用 Python 检查：
  ```bash
  python -c "import pandas as pd; df=pd.read_csv('out/profile_tables/pilot_smoke_measured_profiles.csv'); print(df['profile'].unique())"
  ```
  预期输出应包含 `full_gpu`、至少两个 `kivi` 变体和至少两个 `h2o` 变体。

---

### Task 3: 运行仅限 5 个策略的 Policy 执行

**Files:**
- No code changes; execution via `run_experiment.py`
- Consumes: 新 Profile 表 + 限定后的配置
- Produces: `out/policy_tables/pilot_smoke_measured_policy.csv` 和 summary

- [ ] **Step 1: 清理旧的 Policy 表**  
  ```bash
  rm -f out/policy_tables/pilot_smoke_measured_policy.csv
  rm -f out/policy_tables/pilot_smoke_measured_summary.csv
  ```

- [ ] **Step 2: 执行一体化 Pilot 脚本**  
  ```bash
  python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
  ```
  注意观察 `run_policies` 阶段的输出，确保没有 `unknown policy` 或 `missing measurement` 报错。

- [ ] **Step 3: 检查 Policy 输出表是否只包含 5 个策略**  
  ```bash
  python -c "import pandas as pd; df=pd.read_csv('out/policy_tables/pilot_smoke_measured_policy.csv'); print(df['policy'].unique())"
  ```
  预期结果：`['full_lru' 'static_best' 'static_safe' 'utility_dynamic' 'uncalibrated_dynamic']`。如果出现 `tailguard` 或 `quality_oracle`，说明配置未被正确应用，需回到 Task 1 排查。

---

### Task 4: Baseline Smoke 表验证（H0 / H1 / H2-lite 前置检查）

**Files:**
- Read: `out/policy_tables/pilot_smoke_measured_summary.csv`
- Validate: 各策略的 `violation_rate`, `exact_fallback_ratio`, `p95_ttft_ms`

**Interfaces:**
- 对照设计文档 0.7.3 与 0.7.5 的理论预期

- [ ] **Step 1: 读取 Summary 并验证 `uncalibrated_dynamic` 的违例率**  
  运行 Python 脚本：
  ```python
  import pandas as pd
  df = pd.read_csv("out/policy_tables/pilot_smoke_measured_summary.csv")
  uncal = df[df['name'] == 'uncalibrated_dynamic']
  print(uncal[['epsilon', 'delta', 'violation_rate']])
  ```
  对于 `epsilon=0.05, delta=0.05`，预期 `violation_rate` 显著高于 `delta`（例如 > 0.10）。如果它≤0.05，说明该策略实际上变得保守（可能 bug 导致 fallback 过多）。

- [ ] **Step 2: 验证 `static_best` 的平均质量 vs 尾部质量**  
  提取 `static_best` 的 `mean_quality_loss` 和 `p95_quality_loss`，两者差距应明显（p95 >> mean），证明“平均掩盖尾部”现象（H0 成立）。

- [ ] **Step 3: 验证 `full_lru` 和 `static_safe` 的精确回退率**  
  `full_lru` 的 `exact_fallback_ratio` 应始终为 1.0（因为只用 full）。  
  `static_safe` 的 `exact_fallback_ratio` 应为 0（因为固定 Profile 无精确回退逻辑）。  
  若数据不符，检查 `exact_profiles` 集合是否正确配置。

- [ ] **Step 4: 验证 `utility_dynamic` 的决策分布**  
  检查 `action_distribution` 字段，确认它确实在多种 Profile（如 `kivi_4bit` 和 `h2o`）之间做了选择，而不是卡死在单一 Profile 上。  
  如果全部是 `full_gpu`，说明权重参数（`loss_weight` 和 `memory_weight`）设置不当。

- [ ] **Step 5: 生成 Smoke 表诊断截图（供组会使用）**  
  在 `out/policy_tables/` 下创建一个简短的 `SMOKE_DIAGNOSTIC.txt`，记录：
  - `uncalibrated_dynamic` 的实际 violation_rate
  - `full_lru` vs `utility_dynamic` 的 p95 TTFT 对比
  - 各策略是否完成运行（无报错）
  - 这个文件就是 08-15 里程碑的“交付物”之一。

---

### Task 5: 提交最终交付物（Smoke 表 + 诊断记录）

**Files:**
- Modify: `docs/status/2026-08-15_baseline_smoke_status.md`（新建）
- Add: `out/policy_tables/SMOKE_DIAGNOSTIC.txt`

**Interfaces:**
- N/A

- [ ] **Step 1: 编写状态报告**  
  内容应包括：
  - 5 个策略是否全部顺利跑完
  - H0 是否被 `static_best` 的数据佐证
  - H1 的校准前基线（`uncalibrated`）是否如预期失效
  - H2-lite 的参考基线（`utility_dynamic` vs `full_lru`）的初步对比结果
  - 明确写出“Smoke 表获取任务已完成，可进入下一阶段（实现 TailGuard / Oracle）”或“需回退修复 X 问题”。

- [ ] **Step 2: 提交所有变更**  
  ```bash
  git add docs/status/2026-08-15_baseline_smoke_status.md out/policy_tables/SMOKE_DIAGNOSTIC.txt
  git commit -m "docs: baseline smoke table diagnostic for 5 policies"
  ```

---

## Self-Review

**Spec coverage:**
- 设计文档 0.7.3 要求 5 个 Baseline -> Task 1 配置明确。
- 0.7.4 要求一次性输出 H0/H1/H2-lite -> Task 4 提取诊断数据。
- 0.8 第一阶段要求“可运行的 `run_experiment.py` 和至少 5 个 baseline 的同表对比” -> Task 3 执行，Task 4 验证。
- 所有输出文件路径符合规范。

**Placeholder scan:**
- 所有命令具体可执行，无“TBD”或“后续补充”。
- 验证逻辑给出了明确的 Python 代码片段。

**Type consistency:**
- `config_policies` 返回列表，与 `build_policies` 签名匹配。
- `MetricCollector.summarize_policy_runs` 需要的参数（`epsilon`, `delta`, `exact_profiles`）已在配置中定义，不影响本次执行。

**No missing tasks:** 从配置修改、数据生成、策略执行到诊断报告，闭环完整。
