# Session27 online baseline 交接

## 当前状态

- 工作分支：`main`（2026-09-06 已把 session27 代码与本文档整合归档到 main，本地/远端只保留 main）
- 工作目录：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV`
- 原始 `main` 工作目录遗留的未提交改动（2026-09-05 压力修复中间状态）已备份、未并入 main，见文末「归档与仓库整合记录」。
- 全局约束：session27 输出必须带 `diagnostic_only=true`；质量与 violation 均为 `risk_evidence_insufficient`，不能作为 tail SLO 结论；随机种子固定为 `20260906`；预算 B 取 full、无驱逐扫描的 P25/P50/P75/P90；总体 P95 必须由合并后的逐轮记录计算，不能平均 batch P95。

### 本次恢复状态（2026-09-06 后续交接）

- Task 6 完整 27-session 流程已启动：`out/session27_online_20260906_184717`（9 批 profile-only 分批测量 → merged → budgets → 四档 B sweeps → 聚合与 acceptance）。当前按用户指示视作运行完毕但尚未核对输出；后续需检查该 run root 下 `acceptance.json`、`policy_tables/`（四档 B CSV、`session27_total_summary.csv`、`session27_events.csv`、`baseline_smoke.md`、四张图）并记录 B 敏感性/事件分解结论。运行产物均不提交。
- Task 6 Step1（2-session、单 B、五策略 online smoke）已通过：`out/session27_task6_step1/run_online_smoke/acceptance.json` passed。9 批 profile 合并成功、budgets 生成（2-session 下 P25=4.40234 MiB），单档 B sweep returncode 0，五策略 CSV 行均 `backend_name=online_qwen`、`measured=true`、含 TTFT 与 backend 事件、无回放；四张 `summary_policy_*.png` 与 total summary/events/baseline_smoke.md 齐备。quality/violation 均为 `risk_evidence_insufficient`（图表相应标注或占位）。B 对 TTFT/backend 事件不敏感，结论按事件分解记录，不调整策略制造预设排序。
- Task 5（merged per-turn 汇总与四图）已完成并提交：`a105311 fix: aggregate policy metrics from turn records`。`run_util/session_aggregation.py` 提供 `aggregate_policy_csvs`/`summarize_cells`/事件 CSV/session 点 CSV/markdown 输出，bootstrap CI 种子 `20260906`；`visual/plot_summary.py` 增加 CI 误差带、`risk_evidence_insufficient` 标注与散点，并支持空指标时补占位图（`always_emit_all`）。全量 pytest `447 passed, 3 subtests passed`。
- Task 0（兼容性）已闭环：四个 labeling 源脚本迁回仓库 `scripts/`（`common.py`、`label_longbench_quality.py`、`label_sharegpt_sessions.py`、`prepare_sharegpt_session_candidates.py`，原样复制自备份目录），`tests/test_longbench_label_strategy.py` 与 `tests/test_sharegpt_labeling_strategy.py` 改为从仓库 `scripts/` 导入；全量 pytest 447 passed（含 Task5 新增用例）。
- Task 0 2-session online GPU 冒烟通过（`out/task0_gpu_smoke/`，未提交）：`OnlineQwenSessionBackend` + persistent worker、设备 0/1、B=1024、`max_new_tokens=1`；两 session 第二轮均 `cache_reused=true`，TTFT 真实且无 profile 表回放（`replay_source` 为空）；s1 首轮含模型加载约 40.4 s、复用轮 110.7 ms，s2 首轮 120.6 ms、复用轮 104.3 ms；进程退出 0。

- Task 4 follow-up 已提交：`f09fa94 fix: make policy csv provenance reliable and shadow audits observed-only`（基于 `d67f8b1`）。
- 独立复审 PASS（两轮：初审 With fixes → 复审 Ready to commit Yes，无 Critical/Important）。
- 完整验证：`pytest -p no:cacheprovider -q -k 'not test_qwen2_profile_measurement_appends_pythonpath'` → `435 passed, 1 deselected, 3 subtests passed`。
- 进度账本 `.superpowers/sdd/2026-09-06-session27-online-baselines/progress.md` 已更新 Task 4 complete。
- 工作区本轮不再持有 Task 4 中间未提交改动；`tests/test_experiment_semantics.py` 新增真实 CSV 输出 provenance 回归测试。
- 计划文件、交接文档和 `out/profile_tables/` 仍是未跟踪运行/规划产物，不应提交。

## 已完成并验证

### 1. 分层切分与预算扫描

提交：`1b4ebb6`、`6c5a6d6`、`e2330ec`。

- `split_measurements` 支持按 session 的 50/50 分层切分，固定种子为 `20260906`。
- 同一 session 不会同时进入 calibration 和 evaluation；分层键包含任务、长度桶和最大 turn depth。
- 新增 full、无驱逐 occupancy 扫描，生成 P25/P50/P75/P90 预算 JSON。
- session27 主运行在本次 profile 输出完成后，于同一 run 目录自动派生预算，不依赖未跟踪的固定路径 JSON。
- 已验证：`pytest -q tests/test_policy_session_budget.py tests/test_experiment_semantics.py`，`33 passed`。

### 2. 持久 Qwen 在线 session backend

提交：`d375c0c`、`c660bad`、`c4fe5d7`、`19e11b5`。

- 新增 `OnlineQwenSessionBackend`，`online_qwen` 通过持久 Qwen worker 执行 fixture 请求，结果不再从 profile 表回放 TTFT；`measured_replay` 仍保留为兼容入口。
- backend 维护全局 LRU resident 索引，记录 resident before/after、全局 resident、预算驱逐、恢复、重算和 queue 事件。
- profile 不兼容切换会释放旧 runtime 的 resident KV，并将重算/切换耗时写入事件与结果。
- online session history 由实际先前输出构造；复用 KV 时已处理 prefix-aware attention mask 与 cache position。16-token 回归覆盖 decode position `15..29`。
- worker、运行时或驱逐控制失败会标记 `worker_state_lost`，关闭 worker 并使逻辑 resident 状态失效；控制返回的 `evicted_sessions` 必须完整确认。
- policy CSV 已写入 `diagnostic_only`、`quality_status=risk_evidence_insufficient` 和 `violation_status=risk_evidence_insufficient`。
- 已验证：`pytest -q tests/test_qwen_session_backend.py tests/test_policy_session_budget.py`，`37 passed`。两 turn GPU smoke 已给出第二轮 `cache_reused=true`、TTFT `83.28 ms` 的解析结果。尚未补跑 16-token GPU smoke；有 16-token 的确定性回归测试。

### 3. 五条 baseline 策略语义

提交：`84fdd6c`、`7906bde`、`2d12cfb`、`a02ed7b`。

- `static_best` 使用 calibration 合并样本的 P95 TTFT 点估计；lossy 只有比 full 快至少 5% 时才可被选中。
- `static_safe` 先保留经验 quality-loss 最低的 lossy 主候选；conformal 不安全时最终执行 full，同时保留 primary/fallback 信息。
- `uncalibrated_dynamic` 在满足 `pred_loss <= epsilon` 的 lossy profile 中选择预测 TTFT 最小项。
- `utility_dynamic` 改为全局 primal-dual：`ttft_norm + 0.25*kv_norm + lambda*(loss-epsilon)`，并按 `lambda=max(0, lambda + 0.5/sqrt(t)*residual)` 更新。
- 补充修复：`static_best` 仅在 full/lossy merged calibration P95 都有限且 lossy 快至少 5% 时可选 lossy；`None`、缺失、NaN、正负无穷 TTFT 都 fail-closed 到 exact。registry 会明确拒绝已弃用的 `utility_dynamic.memory_weight` / `loss_weight`，避免静默改变旧配置语义。
- 完成独立审查，未发现 Critical/Important/Minor。独立验证：`pytest -p no:cacheprovider -q tests/test_policy_semantics.py tests/test_policy_session_budget.py tests/test_tailguard_core.py -k 'not test_qwen2_profile_measurement_appends_pythonpath'`，`240 passed, 1 deselected, 3 subtests passed`。

## 原始实施计划与后续顺序

### 4. Shadow audit 与逐轮质量记账

修改 `run_util/core_types.py`、`run_util/run_policies.py`、`run_util/experiment_summary.py`，并新增质量审计测试。

- `PolicyRunRecord` 增加 `audit_selected`、`predicted_quality_loss`、`observed_quality_loss`、`quality_estimate`、`primary_profile`。
- 固定种子下 10% audit 选择应稳定；审计轮用实际 loss，非审计轮用预测；估计采用 inverse-probability 校正。
- utility 的 lambda 使用校正后的质量残差更新；static_safe 的 full fallback 计入最终 TTFT、KV 与质量。

#### 当前状态（已完成并提交）

- 基础提交 `d67f8b1` + follow-up `f09fa94`（PolicyRunRecord audit/quality/primary 字段、固定 `20260906` 的 10% evaluation-population 抽样、inverse-probability 估计、utility lambda 校正、static_safe fallback 记账）。
- follow-up 修复项：`run_policies` 真实 CSV 输出每行带可靠 `config`/`run_dir` provenance（含 `policy_tables` 路径的 run_dir 推导与端到端回归测试）；online audit 用独立 full-profile shadow worker（独立 worker/state、`reset_history=False`、不复用 serving resident KV），审计后逐 session 驱逐 shadow KV，失败时关闭 worker 以限制显存；`audit_sample_count` 只计 observed audit 行；decide/execute 失败记录保留 audit 标记；wide-sweep 聚合拒绝缺 provenance 的 diagnostic CSV 与混合 provenance 目录；total policy summary 行继承 diagnostic 字段。
- 复审：两轮独立审查，最终 Ready to commit Yes（无 Critical/Important；Minor 项记录：shadow 失败仍静默仅靠 audit_sample_count 观察、vLLM adapter 无 persistent_worker 兼容、相对 loss 与 pred_loss 尺度差异需在真实 lossy calibration 复核）。
- 验证：`pytest -p no:cacheprovider -q -k 'not test_qwen2_profile_measurement_appends_pythonpath'` → `435 passed, 1 deselected, 3 subtests passed`。
- 进度账本已更新 Task 4 complete。可以开始 Task 5。

### 5. 正确汇总与四张图

当前状态：已完成（`a105311`），见上方恢复状态。


修改 `run_util/experiment_summary.py` 和 `visual/plot_summary.py`，并新增 session summary 测试。

- 对 merged per-turn records 计算每个 `(B, epsilon, delta, policy)` 的总体 P95、均值、动作分布和 backend 事件。
- 使用 session block bootstrap 输出 95% 误差条；static_best 的选择仍使用点估计。
- 保留四张 `summary_policy_*.png`。quality/violation 图明确标记 `risk_evidence_insufficient`。
- 新增事件汇总 CSV，联读 TTFT、budget hit、restore、recompute、queue。

### 6. 端到端诊断与验收

当前状态：Step1 冒烟通过；完整 27-session 流程已在 `out/session27_online_20260906_184717` 启动并视作运行完毕（输出待核对）。


修改 `scripts/run_diagnostic_session_batches.py`、配置和交接文档。

1. 先跑 2-session、单 B、五策略 online smoke，确保没有 replay source，且每轮都有 backend event 字段。
2. 再异步执行 session27 的四档 B；只有每一批完整成功后才合并结果。
3. 验收 B 至少在一个 backend 事件指标上有变化；策略切换能看到 restore 或 recompute；所有输出保留 diagnostic/risk provenance。
4. 若 `full_lru` 仍最快或 B 对 TTFT 不敏感，报告事件分解，不调整策略来制造预设排序。

## 恢复执行

> 以下为归档前在 session27 工作树的执行记录；归档后统一在 `main` 继续（工作目录 `/work/TailGuardKV`）。原 session27 工作树路径仅作历史参考。

- 计划文件：`docs/superpowers/plans/2026-09-06-session27-online-baselines.md`
- 进度账本：`.superpowers/sdd/2026-09-06-session27-online-baselines/progress.md`
- Task 3 报告：`.superpowers/sdd/2026-09-06-session27-online-baselines/task-3-report.md`
- Task 4 报告：`.superpowers/sdd/2026-09-06-session27-online-baselines/task-4-report.md`
- Task 4 已提交（`d67f8b1` + `f09fa94`）、复审 PASS、账本更新。下一步是 Task 5（正确汇总与四张图），再 Task 6（GPU 端到端，需先确认环境有 GPU 与 fixture/模型）。
- Task 5 要点：merged per-turn records 上按 `(B, epsilon, delta, policy)` 算总体 P95/均值/动作分布/backend 事件（不可平均 batch P95）；session block bootstrap 输出 95% 误差条（种子 `20260906`，static_best 选择仍用点估计）；保留四张 `summary_policy_*.png`，quality/violation 图显式标记 `risk_evidence_insufficient`；新增事件汇总 CSV。先写失败测试，再做实现并提交 `fix: aggregate policy metrics from turn records`。
- 全量测试的已知环境失败：`tests/test_tailguard_core.py::TailGuardCoreTest::test_qwen2_profile_measurement_appends_pythonpath` 将 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV/third_party/H2O/h2o_hf` 写死；本工作树位于 `TailGuardKV-session27-online`。不要把它误判为回归。
- 不要提交 `out/profile_tables/` 或 `.superpowers/` 运行产物（运行产物已移入备份目录）。根目录本交接文档与计划文件已在 2026-09-06 归档提交进 `main`。


## 归档与仓库整合记录（2026-09-06）

本轮只做仓库整合与清理，不改业务代码：

- session27-online 全部代码改动此前已合入 `main` 并推送（`origin/main` = `f09fa94`，435 passed）。本交接文档与实施计划已在此提交归档进 `main`；后续 Task 5/6 直接在 `main` 上继续。
- 原 main 工作目录遗留的未提交改动（31 个 tracked 文件变更、删除 28 个旧 tests、若干未跟踪运行产物）已原样备份：`/DATACENTER3/zhenxiang.wang/backups/20260906-tailguardkv-cleanup/main-worktree/`（含 `main-worktree-HEAD-fcca1f9.patch`）。Task 6 如需其中的 batch/压力改动，可从备份取回再合。
- `TailGuardKV-external-baseline` 与 `TailGuardKV-session27-online` 是本仓库的两个 git worktree（非独立仓库）；`TailGuardKV-labeling` 是非 git 的独立标注目录。三者的未跟踪运行产物与账本已备份到上述备份目录；worktree 与兄弟目录随后移除。
- 未并入 main 的分支以 annotated tag 归档后删除：
  - `archive/external-baseline-flow-20260906`：`feat/external-baseline-flow`（含 WIP 改动与 artifacts 的归档提交）
  - `archive/sdd-police-refactor-20260906`：`sdd/police-refactor@c6413c1`
  - `archive/pilot-profile-bugfix-origin-20260906`：`origin/pilot-profile-bugfix@b70f4aa`（与当前仓库无共同祖先的旧历史）
- 已完整合入 main 的 `feat/session27-online-backend` 与本地 `pilot-profile-bugfix` 直接删除。本地/远端最终只保留 `main`。
