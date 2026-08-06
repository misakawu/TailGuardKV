# 显存预算扫描实验执行计划（基于用户选择）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 3000–5000 MiB 范围内以 500 MiB 步幅（5 个预算点）× 3 组 ε/δ 组合（共 15 个实验单元）执行一体化 Pilot 实验，覆盖 5 个 Baseline 策略，生成用于分析“显存预算 vs 系统性能/违例率”关系的完整数据集。

**Architecture:** 复用现有 `run_experiment.py` 入口，通过 `--memory-budget-mib`、`--epsilon`、`--delta` 命令行参数逐点驱动。Profile 表复用现有文件，不重新构建。通过 `nohup` + `conda run -n tailguardkv-base` 包装执行，`CUDA_VISIBLE_DEVICES=0,1` 指定两张显卡。

**Tech Stack:** Python 3.10+, Conda, Bash, run_experiment.py, CUDA 11.8+.

---

## Global Constraints

- **Conda 环境**：`tailguardkv-base`（路径：`/DATACENTER3/zhenxiang.wang/miniforge3/envs/tailguardkv-base`）
- **GPU**：`CUDA_VISIBLE_DEVICES=0,1`
- **Profile 表**：复用 `out/profile_tables/pilot_smoke_measured_profiles.csv`，不重新构建
- **预算列表**：`[3000, 3500, 4000, 4500, 5000]` MiB
- **ε/δ 组合**：
  - `(ε=0.05, δ=0.05)`
  - `(ε=0.1, δ=0.05)`
  - `(ε=0.1, δ=0.1)`
- **策略集合**：5 个 Baseline（`full_lru`, `static_best`, `static_safe`, `utility_dynamic`, `uncalibrated_dynamic`）
- **每点每策略**：100 条请求（`measured-replay` Backend）
- **输出目录**：每个预算点独立时间戳目录
- **日志**：完整输出到 `nohup.out`（或指定日志文件）
- **通知**：手动检查，无邮件/标记文件
- **预计总时长**：5 预算 × 3 ε/δ × ~10 分钟 ≈ 150 分钟（2.5 小时）

---

### Task 0: 创建预算扫描执行脚本

**Files:**
- Create: `scripts/run_budget_scan.sh`
- Modify: `configs/pilot.yaml`（确保 `policies.names` 仅含 5 个 Baseline）

**Interfaces:**
- Consumes: `configs/pilot.yaml`, `out/profile_tables/pilot_smoke_measured_profiles.csv`
- Produces: 可执行的 bash 脚本，包含 15 个实验单元的循环

- [ ] **Step 1: 确认配置文件仅含 5 个 Baseline**  
  检查 `configs/pilot.yaml` 中 `policies.names`，确保只包含 `full_lru`, `static_best`, `static_safe`, `utility_dynamic`, `uncalibrated_dynamic`。若有 `tailguard` 或 `quality_oracle`，先注释或删除。

- [ ] **Step 2: 创建 `scripts/run_budget_scan.sh`**  
  写入以下伪代码逻辑：

```
脚本头部：
  设置 conda 环境变量（指向 tailguardkv-base）
  设置 CUDA_VISIBLE_DEVICES=0,1
  定义预算数组 BUDGETS = [3000, 3500, 4000, 4500, 5000]
  定义 ε/δ 组合数组 PAIRS = [(0.05,0.05), (0.10,0.05), (0.10,0.10)]
  定义配置文件路径 = configs/pilot.yaml
  定义 Profile 表路径 = out/profile_tables/pilot_smoke_measured_profiles.csv
  创建日志目录 out/budget_scan_logs

主循环：
  对于每个 budget in BUDGETS:
    对于每个 (eps, delta) in PAIRS:
      构建实验标签 = "budget_{budget}_eps_{eps}_delta_{delta}"
      定义日志文件 = out/budget_scan_logs/{label}.log
      记录开始时间到日志

      执行命令：
        conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured \
          --config 配置文件 \
          --memory-budget-mib {budget} \
          --epsilon {eps} \
          --delta {delta} \
          --measurements Profile表路径

      捕获命令退出码
      记录结束时间和退出码到日志
      如果退出码为 0，标记 SUCCESS，否则标记 FAILED
      累加成功/失败计数器

汇总：
  打印总实验数、成功数、失败数
  如果存在失败，退出码为 1，否则为 0
```

- [ ] **Step 3: 赋予脚本执行权限**  
  `chmod +x scripts/run_budget_scan.sh`

- [ ] **Step 4: 验证脚本语法**  
  用 bash 的语法检查模式验证脚本无误。

---

### Task 1: 执行实验（nohup + conda run）

**Files:**
- No code changes; execution step.
- Monitors: `tail -f out/budget_scan_logs/*.log`

- [ ] **Step 1: 切换到项目根目录**  
  `cd /DATACENTER3/zhenxiang.wang/tailguardkv`

- [ ] **Step 2: 验证 conda 环境可用**  
  执行 `conda run -n tailguardkv-base python --version`，确认返回 Python 3.10+。

- [ ] **Step 3: 验证 CUDA 可用性**  
  执行 `conda run -n tailguardkv-base python -c "import torch; print(torch.cuda.is_available())"`，确认返回 `True`。

- [ ] **Step 4: 用 nohup 启动实验**  
  执行：`nohup ./scripts/run_budget_scan.sh > nohup.out 2>&1 &`  
  将进程 PID 写入 `out/budget_scan.pid`

- [ ] **Step 5: 确认进程已启动**  
  通过 PID 文件检查进程是否在运行。

- [ ] **Step 6: 查看实时日志（可选）**  
  用 `tail -f nohup.out` 或 `tail -f out/budget_scan_logs/*.log` 监控进度。

---

### Task 2: 监控实验进度

**Files:**
- Read: `nohup.out`, `out/budget_scan_logs/*.log`

- [ ] **Step 1: 查看整体进度**  
  查看 `nohup.out` 末尾，确认当前正在运行的实验标签。

- [ ] **Step 2: 检查是否所有实验都完成了**  
  统计 `out/budget_scan_logs/` 下 `.log` 文件数量，预期最终为 15。

- [ ] **Step 3: 检查是否有失败实验**  
  在所有日志文件中搜索 "FAILED" 字符串。

- [ ] **Step 4: 检查主进程状态**  
  通过 PID 文件检查 `run_budget_scan.sh` 进程是否仍在运行。

---

### Task 3: 汇总实验结果

**Files:**
- Create: `scripts/aggregate_budget_results.py`
- Create: `out/budget_scan_summary.csv`

- [ ] **Step 1: 创建汇总脚本 `scripts/aggregate_budget_results.py`**  

伪代码逻辑：

```
导入必要库 (pandas, re, pathlib, glob)

定义日志目录 = out/budget_scan_logs
定义输出文件 = out/budget_scan_summary.csv

定义正则表达式，从日志文件名解析 budget, eps, delta

初始化空列表 rows

对于每个日志文件 in 日志目录:
  用正则匹配文件名，提取 budget, eps, delta
  如果匹配失败，跳过该文件
  读取日志文件内容
  在日志中查找输出目录路径（匹配 "Output directory: out/..."）
  如果找不到，跳过该文件
  构建 summary 文件路径：{输出目录}/policy_tables/pilot_smoke_measured_summary.csv
  如果 summary 文件不存在，跳过
  用 pandas 读取 summary CSV
  筛选 section == "policy" 的行
  对于每一行，提取：
    policy 名称
    violation_rate
    p95_ttft_ms
    mean_quality_loss
    p95_quality_loss
    mean_kv_cache_memory_mib
    p95_peak_memory_mib
    exact_fallback_ratio
    action_distribution
    mean_ttft_ms
    p99_ttft_ms
  将 (budget, eps, delta, 各指标) 组合成一行，加入 rows 列表

如果 rows 不为空：
  转换为 DataFrame
  写入 CSV 文件
  打印汇总统计（总记录数、预算点数量）
否则：
  打印警告：未找到任何有效数据
```

- [ ] **Step 2: 执行汇总脚本**  
  用 conda run 执行 `scripts/aggregate_budget_results.py`

- [ ] **Step 3: 检查汇总结果**  
  查看 `out/budget_scan_summary.csv` 的前几行，确认列结构完整。

- [ ] **Step 4: 生成可视化图表（可选）**  
  创建绘图脚本，伪代码逻辑：
```
读取汇总 CSV
筛选固定 (eps=0.05, delta=0.05) 的数据
对每个 policy：
  按 budget 排序
  绘制 p95_ttft_ms vs budget 曲线，带标记点
添加图例、坐标轴标签、标题
保存为 PNG 文件
```

---

### Task 4: 最终交付物

- [ ] **Step 1: 确保所有 15 个实验完成**  
  检查 `out/budget_scan_logs/` 下有 15 个 `.log` 文件，且每个都包含 `SUCCESS` 或至少没有 `FAILED`。

- [ ] **Step 2: 复制关键数据到可读位置**  
  将汇总 CSV 复制为带时间戳的备份。

- [ ] **Step 3: 编写简短的状态报告**  
  在 `docs/status/2026-08-06_budget_scan_complete.md` 中记录：
  - 实验参数（预算范围、ε/δ 组合）
  - 运行时长
  - 成功/失败统计
  - 初步观察（如：`utility_dynamic` 在哪个预算点开始显著优于 `full_lru`，`uncalibrated_dynamic` 违例率随预算变化的趋势）

---

## Self-Review

**用户选择确认：**

| 问题 | 用户选择 | 脚本实现 |
| :--- | :--- | :--- |
| Q1 Conda 环境 | `tailguardkv-base`，GPU 0,1 | ✅ `CUDA_VISIBLE_DEVICES=0,1` + `conda run -n tailguardkv-base` |
| Q2 Profile 表 | 复用现有 | ✅ 脚本中检查文件存在，不重新构建 |
| Q3 输出目录 | 每个预算点独立 | ✅ `run_experiment.py` 默认行为 |
| Q4 日志保留 | 完整日志 | ✅ 每个实验单元独立 `.log` 文件 + `nohup.out` |
| Q5 通知方式 | 手动检查 | ✅ 无邮件/标记文件 |
| Q6 ε/δ 范围 | 3 组组合 | ✅ 已定义三组参数 |

**注意事项：**
- 需确认 `run_experiment.py` 是否支持 `--measurements` 参数传入外部 Profile 表路径。若不支持，可通过 `configs/pilot.yaml` 的 `outputs.smoke_profiles` 指定。
- Profile 表复用前提：当前目录已有该文件，`run_experiment.py` 会优先使用。
- 日志文件较多（15 个），每个约 1–2 MB，总磁盘占用可接受。

---

**计划完成。执行命令：**

```bash
cd /DATACENTER3/zhenxiang.wang/tailguardkv
nohup ./scripts/run_budget_scan.sh > nohup.out 2>&1 &
echo $! > out/budget_scan.pid
```