# Baseline Smoke 表生成实现计划

> **对于执行者：** 必须使用子技能 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步实现。任务使用复选框 `- [ ]` 标记跟踪。

**目标：** 修补现有代码库，使 `run_experiment.py pilot-smoke-measured` 能够成功生成包含全部 7 个策略（full_lru、static_best、static_safe、utility_dynamic、uncalibrated_dynamic、tailguard、quality_oracle）的 Baseline Smoke 总表，并正确报告违例率、p95 TTFT、控制器开销（QRP/CG 耗时）和精确回退率。

**架构：** 现有的三层架构（Profile / Policy / Backend）保持不变。缺失的是策略注册（registry）和两个具体策略类（TailGuard、QualityOracle）。`StatsPolicy` 基类已提供预测器、保形门和候选过滤，`TailGuardPolicy` 将在其上增加按请求的计时与安全集合选择；`QualityOraclePolicy` 则利用真实损失选择最优动作。`MetricCollector` 已具备汇总所需指标的能力。

**技术栈：** Python 3.10+，PyTorch，vLLM/LMCache（抽象层），pandas（可选），yaml。

---

## 全局约束

- Python 版本 ≥ 3.10。
- 所有策略必须实现 `decide(request, cache_state, device_state) -> Action`。
- 控制器开销字段（`controller_qrp_ms`、`controller_cg_ms`、`controller_overhead_ms`）必须通过 `Action` 上报。
- `quality_oracle` 策略须设置 `oracle = True`，`placeholder = False`。
- `tailguard` 策略须使用 `guard.risk_upper()` 和 `guard.is_safe()` 过滤动作，安全集合为空时回退到精确 profile。
- 配置文件 `configs/pilot.yaml` 中的 `policies.names` 必须包含全部 7 个策略名称。
- 执行命令：`python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml`。
- 输出路径：`out/profile_tables/pilot_smoke_measured_profiles.csv`、`out/policy_tables/pilot_smoke_measured_policy.csv`、`out/policy_tables/pilot_smoke_measured_summary.csv`。

---

### 任务 1：在 `policies/registry.py` 中注册缺失的策略类

**文件：** `policies/registry.py`（修改）

**模块说明：**  
该模块负责根据配置的策略名称列表，实例化对应的策略对象。目前缺少 `tailguard` 和 `quality_oracle` 的映射分支。本任务添加这两个分支，并在文件头部导入对应的类（类将在任务 2、3 中实现）。

**接口：**
- 输入：策略名称列表 `names`，校准测量 `calibration_measurements`，全量测量 `oracle_measurements`，profiles 列表，epsilon，delta，exact_profiles，memory_budget_mib。
- 输出：`list[Policy]`。

**步骤：**

- [ ] **步骤 1：导入新策略类**  
  在文件顶部添加：
  ```python
  from policies.tailguard import TailGuardPolicy
  from policies.quality_oracle import QualityOraclePolicy
  ```

- [ ] **步骤 2：在 `build_policies` 循环中添加分支**  
  对于每个 `name`，增加两个 `elif` 分支，分别匹配 `"tailguard"` 和 `"quality_oracle"`。  
  **伪代码：**
  ```
  if name == "tailguard":
      创建 TailGuardPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib)
      加入 policies 列表
  elif name == "quality_oracle":
      创建 QualityOraclePolicy(oracle_measurements, profiles, epsilon, exact_profiles, memory_budget_mib)
      加入 policies 列表
  ```

- [ ] **步骤 3：验证导入**  
  运行 `python -c "from policies.registry import build_policies; print('OK')"`，预期成功（可能因缺失类而报错，但下一步会解决）。

- [ ] **步骤 4：提交**  
  `git add policies/registry.py && git commit -m "feat: 注册 TailGuard 和 QualityOracle 策略"`

---

### 任务 2：实现 `QualityOraclePolicy`

**文件：** 新建 `policies/quality_oracle.py`

**模块说明：**  
该策略是“上界 Oracle”，它预先知道每个请求在每个 profile 下的真实质量损失（`quality_loss`）。它只选择满足 `quality_loss <= epsilon` 且 TTFT 最小的 profile，用于衡量“在完美信息下能达到的最佳系统收益”。它不参与“最佳可部署 Baseline”排名，仅作为比较的上界。

**接口：**
- 构造函数接收 `oracle_measurements: Iterable[ProfileMeasurement]`、`profiles`、`epsilon`、`exact_profiles`、`memory_budget_mib`。
- `decide` 返回 `Action`，其中 `safe=True`。

**伪代码设计：**

```
类 QualityOraclePolicy 继承自 Policy:
    设置 oracle = True
    设置 placeholder = False

    构造函数(测量数据, profiles, epsilon, exact_profiles, 显存预算):
        初始化名字 = "quality_oracle"
        保存 profiles, epsilon, exact_profiles, 显存预算
        建立查找字典 loss_lookup[(request_id, profile)] = quality_loss
        建立查找字典 cost_lookup[request_id][profile] = ttft_ms

    定义 _fallback_profile():
        遍历 profiles，若 profile 在 exact_profiles 中则返回该 profile
        否则返回 profiles[0]

    定义 decide(request, cache_state, device_state):
        设 best_profile = _fallback_profile()
        设 best_score = 无穷大
        对于每个 profile in profiles:
            若 cost_lookup 无此请求或 profile，跳过
            取 cost = cost_lookup[request_id][profile]
            若 cost 无穷大，跳过
            取 loss = loss_lookup[(request_id, profile)]
            若 loss 为 None，跳过
            若 loss <= epsilon 且 cost < best_score:
                更新 best_score = cost
                更新 best_profile = profile
        返回 Action(profile=best_profile, reason="oracle_optimal", pred_loss=0.0, risk_upper=0.0, safe=True, epsilon=epsilon)
```

- [ ] **步骤 1：创建文件并编写类骨架与上述逻辑**
- [ ] **步骤 2：提交**  
  `git add policies/quality_oracle.py && git commit -m "feat: 添加 QualityOraclePolicy"`

---

### 任务 3：实现 `TailGuardPolicy`

**文件：** 新建 `policies/tailguard.py`

**模块说明：**  
这是本论文的核心策略。它继承自 `StatsPolicy`（提供了 `predictor`、`guard`、`_candidate_profiles`、`_ttft_or_inf` 等工具）。  
`decide` 方法依次执行：
1. 对每个有损 profile 调用 `predictor.predict_loss`，并累计 QRP 耗时。
2. 对每个有损 profile 调用 `guard.risk_upper` 判断是否 `<= epsilon`，并累计 CG 耗时。
3. 在安全的有损 profile 中，选择 `_ttft_or_inf` 最小的 profile。
4. 若无安全有损 profile，则回退到精确 fallback profile（质量损失为 0）。
5. 在返回的 `Action` 中填入 `controller_qrp_ms`、`controller_cg_ms` 和总开销。

**接口：**
- 构造函数同 `StatsPolicy`。
- `decide` 返回 `Action`。

**伪代码设计：**

```
类 TailGuardPolicy 继承自 StatsPolicy:
    构造函数(校准测量, profiles, epsilon, delta, exact_profiles, 显存预算):
        调用父类构造函数，名字设为 "tailguard"

    定义 decide(request, cache_state, device_state):
        # 1. QRP 计时
        start_qrp = 当前时间
        候选列表 = []
        有损候选 = [p for p in _candidate_profiles(include_exact=False) if p not in exact_profiles]
        对于每个 profile in 有损候选:
            pred_loss = predictor.predict_loss(request, profile)
            候选列表追加 (profile, pred_loss)
        qrp_ms = (当前时间 - start_qrp) * 1000

        # 2. CG 计时
        start_cg = 当前时间
        安全列表 = []
        对于每个 (profile, pred_loss) in 候选列表:
            risk_upper = guard.risk_upper(request, profile, pred_loss)
            若 risk_upper <= epsilon:
                安全列表追加 (profile, pred_loss, risk_upper)
        cg_ms = (当前时间 - start_cg) * 1000

        # 3. STC：在安全有损中选最小 TTFT
        best_profile = _fallback_profile()
        best_cost = 无穷大
        对于每个 (profile, pred_loss, risk_upper) in 安全列表:
            cost = _ttft_or_inf(profile)
            若 cost < best_cost:
                更新 best_cost, best_profile, best_pred_loss, best_risk_upper

        # 4. 若没有安全有损，回退到精确 fallback
        若 best_profile == _fallback_profile():
            fallback = _fallback_profile()  # 保证是 exact
            返回 Action(profile=fallback, reason="tailguard_exact_fallback",
                       pred_loss=0.0, risk_upper=0.0, safe=True,
                       epsilon=epsilon, delta=delta,
                       fallback_reason="no_safe_lossy_profile",
                       controller_qrp_ms=qrp_ms, controller_cg_ms=cg_ms,
                       controller_overhead_ms=qrp_ms + cg_ms)

        # 5. 返回安全有损最优动作
        返回 Action(profile=best_profile, reason="tailguard_safe_optimal",
                   pred_loss=best_pred_loss, risk_upper=best_risk_upper,
                   safe=True, epsilon=epsilon, delta=delta,
                   fallback_reason="calibrated_safe",
                   controller_qrp_ms=qrp_ms, controller_cg_ms=cg_ms,
                   controller_overhead_ms=qrp_ms + cg_ms)
```

- [ ] **步骤 1：创建文件并编写上述类和方法**
- [ ] **步骤 2：提交**  
  `git add policies/tailguard.py && git commit -m "feat: 添加 TailGuardPolicy 核心逻辑与计时"`

---

### 任务 4：更新配置文件包含全部策略

**文件：** `configs/pilot.yaml`（修改）

**模块说明：**  
配置文件控制实验参数，包括策略列表。当前可能缺少 `tailguard` 和 `quality_oracle`。本任务确保 `policies.names` 或 `policies.items` 中包含全部 7 个名字。

**伪代码设计：**

若文件中有 `policies.names`，则改为：
```
policies:
  names:
    - full_lru
    - static_best
    - static_safe
    - utility_dynamic
    - uncalibrated_dynamic
    - tailguard
    - quality_oracle
```
若使用的是 `items` 结构，则类似修改。

- [ ] **步骤 1：编辑 `configs/pilot.yaml`，添加缺失的策略**
- [ ] **步骤 2：验证 YAML 语法**  
  `python -c "import yaml; yaml.safe_load(open('configs/pilot.yaml'))"`
- [ ] **步骤 3：提交**  
  `git add configs/pilot.yaml && git commit -m "config: 为 pilot 添加 tailguard 和 quality_oracle 策略"`

---

### 任务 5：执行完整 Pilot Smoke 实验

**文件：** 无代码修改，只执行命令。

**模块说明：**  
运行一体化实验脚本，它会先构建 Profile 表（`run_build_profile_table`），再运行所有策略（`run_run_policies`），最后生成汇总 CSV。我们需要检查输出文件是否存在且包含合理数据。

**步骤：**

- [ ] **步骤 1：清理旧输出**  
  `rm -f out/profile_tables/pilot_smoke_measured_profiles.csv out/policy_tables/pilot_smoke_measured_policy.csv out/policy_tables/pilot_smoke_measured_summary.csv`

- [ ] **步骤 2：执行实验**  
  `python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml`  
  观察输出，确保无异常报错。

- [ ] **步骤 3：检查文件生成**  
  确认三个 CSV 文件均存在且大小 > 0。

- [ ] **步骤 4：快速验证汇总表内容**  
  用 Python 读取 summary CSV，检查 `section=='policy'` 的各行，确认：
  - `tailguard` 有 `controller_qrp_ms`、`controller_cg_ms` 非空。
  - `uncalibrated_dynamic` 的 `violation_rate` 明显高于 `delta`（例如 > 0.10）。
  - `tailguard` 的 `violation_rate` ≤ `delta`（例如 ≤ 0.05）。
  - `exact_fallback_ratio` 小于 0.6。

- [ ] **步骤 5：提交运行结果（可选）**  
  可记录 summary 中的关键数值，但不必提交大数据文件。

---

### 任务 6：撰写阶段性 Go/No-Go 决策

**文件：** 新建 `docs/decisions/pilot_go_nogo_20260806.md`

**模块说明：**  
依据设计文档 0.7.5 和 0.8 的要求，汇总本次 Pilot 的结果，判断是否满足全部五项通过标准，并给出明确结论（Go / Conditional Go / No-Go）。

**伪代码内容：**
```
# Pilot 实验结果汇总

日期：2026-08-06
配置：Qwen2.5-7B，2 任务，2 显存预算，epsilon=0.2, delta=0.05

## 五项门槛检查
1. H0 tail-risk 存在：……（引用数据，平均 vs q95 损失差距 ≥2×）
2. H1 calibration 守住违例率：empirical violation ≤ 8%，且 fallback ratio <60%
3. H2-lite 收益：p95 TTFT 相比最佳 baseline 降 ≥10% 或显存降 ≥15%（列出数据）
4. 控制器开销：qrp_ms <1ms，cg_ms <0.1ms（列出实际值）
5. 代码可扩展性：`run_experiment.py` 无需重写即可用于主实验

## 结论
- [ ] 全部通过 → Go
- [ ] 部分通过 → Conditional Go（说明待修复项）
- [ ] 不通过 → No-Go（说明原因及下一步）
```

- [ ] **步骤 1：根据 Task 5 的数据填写上述文档**
- [ ] **步骤 2：提交**  
  `git add docs/decisions/pilot_go_nogo_20260806.md && git commit -m "docs: 记录 Pilot Go/No-Go 决策"`

---
