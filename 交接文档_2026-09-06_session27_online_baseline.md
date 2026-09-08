# Session27 online baseline 交接

更新时间：2026-09-07

## 特殊注意
所有进程应该使用conda环境和nohub+setid方式启动

## Plan 原文


```text
1. 16-token GPU smoke
2. 定位 worker/runtime 显存泄漏或生命周期问题
3. 修复 cleanup 和失败恢复逻辑
4. 增加对应回归测试
5. 补跑 batch001/003/005/007
6. 验证 9 个 batch 完整覆盖
7. 合并 per-turn profile
8. 从本次运行生成 P25/P50/P75/P90
9. 执行四档 B × 五策略 online sweep
10. 生成汇总表、四张图和事件分解
11. 运行 acceptance
12. 更新交接文档
13. 运行测试并提交代码
```


## 已完成的工作

### 1. 分层切分和预算扫描

已完成并提交：`1b4ebb6`、`6c5a6d6`、`e2330ec`。

- calibration 和 evaluation 按 session 做 50/50 分层切分。
- 同一个 session 不会同时出现在两个集合中。
- 分层键包含任务、长度桶和最大 turn depth。
- 随机种子固定为 `20260906`。
- full profile 支持无驱逐 occupancy 扫描。
- 预算应从本次运行生成的 occupancy 结果中取 P25、P50、P75、P90。

### 2. 在线 Qwen session backend

已完成并提交：`d375c0c`、`c660bad`、`c4fe5d7`、`19e11b5`。

- `OnlineQwenSessionBackend` 使用持久 Qwen worker 执行请求。
- online execution 不再从 profile 表回放 TTFT。
- backend 维护全局 LRU resident KV 索引。
- 记录 resident、eviction、budget hit、restore、recompute 和 queue 事件。
- profile 不兼容时会触发重算。
- cache reuse 已处理 prefix-aware attention mask 和 cache position。
- worker 或 runtime 失败时会标记 `worker_state_lost`，并使逻辑 resident 状态失效。
- 输出带有 `diagnostic_only` 和 risk provenance。

已经完成 2-session GPU smoke，并验证过 cache reuse 和真实 TTFT。16-token 的确定性回归测试已有，但 16-token GPU smoke 尚未补跑。

### 3. 五条 baseline 策略

已完成并提交：`84fdd6c`、`7906bde`、`2d12cfb`、`a02ed7b`。

- `static_best` 使用 calibration 合并样本的 P95 TTFT 点估计；lossy 至少快 5% 才能被选中。
- `static_safe` 保留 lossy primary profile；conformal 不安全时最终执行 full，并保留 fallback 信息。
- `uncalibrated_dynamic` 在 `pred_loss <= epsilon` 的 lossy profile 中选择预测 TTFT 最小项。
- `utility_dynamic` 使用全局 primal-dual lambda 更新。
- 非有限或缺失的校准 TTFT 会 fail-closed 到 exact/full。
- registry 会拒绝已废弃的 utility 参数，避免旧配置静默改变含义。

### 4. Shadow audit 和质量记账

已完成并提交：`d67f8b1`、`f09fa94`。

- 每轮记录 audit 标记、预测质量、观测质量、质量估计和 primary profile。
- 固定种子下抽取 10% evaluation 轮做 audit。
- 非 audit 轮使用预测质量，audit 轮使用真实质量。
- 使用 inverse-probability 方法估计质量。
- utility 的 lambda 使用校正后的质量残差。
- `static_safe` 的 full fallback 纳入最终 TTFT、KV 和质量统计。
- online audit 使用独立的 full-profile shadow worker，不复用 serving resident KV。
- policy CSV 保留可靠的 `config` 和 `run_dir` provenance。

### 5. 汇总和可视化

已完成并提交：`a105311`。

- 总体 P95 从合并后的逐轮记录直接计算，不能平均 batch P95。
- 使用 session block bootstrap 计算 95% 误差区间。
- 生成 TTFT、KV memory、quality loss、violation rate 四类图。
- 新增事件汇总、session 点汇总和 baseline smoke 文档。
- quality 和 violation 只能标记为 `risk_evidence_insufficient`，不能解释为 tail SLO 结论。

### 6. 已通过的 online smoke

目录：

```text
out/session27_task6_step1/run_online_smoke
```

这个 2-session、单预算、五策略 smoke 已通过。它验证了：

- 使用的是 `online_qwen` backend。
- 输出标记为 measured，且没有 replay source。
- 每轮都有 backend event 字段。
- 第二轮能够复用 KV。
- 汇总表、事件表、四张图和 `baseline_smoke.md` 均能生成。

最近一次相关测试提交为：`5cd183d test: validate online session27 baseline diagnostics`。


## 尚未完成的任务

### P0：先解决 profile 阶段的 CUDA OOM

1. 补跑 16-token GPU smoke，确认长 decode 下 cache position、attention mask 和显存变化正常。
2. 检查 persistent worker 是否跨 batch 保留旧模型、KV tensor 或 runtime 引用。
3. 检查 profile 切换时是否只清理了逻辑 resident index，实际 GPU tensor 却仍被引用。
4. 检查 worker shutdown、runtime teardown、`torch.cuda.empty_cache()` 和进程边界的调用时机。
5. 检查双 GPU 的显存分配是否长期偏向 GPU 1，以及是否有 allocator fragmentation。
6. 不要把 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 当成根因修复。它最多用于辅助验证。
7. 增加 worker/runtime cleanup、OOM 后失效、fresh worker 重启和 batch 完成后释放的回归测试。
8. 修复后重新执行失败的四个 batch：`batch001`、`batch003`、`batch005`、`batch007`。

建议重点查看：

```text
backends/qwen_session.py
profiles/qwen2_kv_runtime.py
scripts/run_session27_online_sweeps.py
scripts/run_diagnostic_session_batches.py
```

每个重新执行的 batch 都必须满足：

```text
profile_rows == 120
returncode == 0
mergeable == true
errors == []
```

### P1：完成 profile 合并和预算生成

所有 9 个 batch 成功后：

1. 检查每个 request/profile 组合恰好一行。
2. 检查没有重复、缺失或残留失败记录。
3. 检查 `diagnostic_only=true`、`measured=true`，且没有 replay source。
4. 检查所有 CSV 的 `config` 和 `run_dir` provenance 一致。
5. 只合并本次运行的完整 batch，不能混入旧目录或 `out/profile_tables/`。
6. 生成本次运行自己的：

```text
merged/profile_tables/diagnostic_session27_profiles.csv
```

7. 使用本次 merged full profile 的无驱逐 occupancy 结果生成 P25/P50/P75/P90 四档 B。

### P1：运行四档 B 的五策略 online sweep

对 P25、P50、P75、P90 四个预算，分别运行：

- `full_lru`
- `static_best`
- `static_safe`
- `uncalibrated_dynamic`
- `utility_dynamic`

每档预算都必须使用同一份完整 merged 输入和 online evaluation fixture，并保留真实 TTFT、backend event、audit 字段以及 diagnostic/risk provenance。

检查重点：

- 小预算是否出现 `budget_hit` 或 eviction。
- 策略切换是否产生 `restore` 或 `recompute`。
- shadow audit 是否没有污染 serving resident KV。
- `static_safe` fallback 是否计入最终指标。
- utility lambda 是否使用校正质量估计。

### P2：汇总、验收和结论

重新生成并检查：

```text
policy_tables/session27_total_summary.csv
policy_tables/session27_events.csv
policy_tables/session27_session_points.csv
policy_tables/baseline_smoke.md
policy_tables/summary_policy_p95_ttft.png
policy_tables/summary_policy_kv_memory.png
policy_tables/summary_policy_quality_loss.png
policy_tables/summary_policy_violation_rate.png
```

验收条件：

1. 总体 P95 来自 merged per-turn records，而不是 batch P95 的平均值。
2. 95% 误差区间使用 session block bootstrap。
3. 至少一个 backend 事件指标随 B 变化，例如 `budget_hit_count`、`evicted_kv_mib`、resident KV、restore 或 recompute。
4. 能看到真实的 cache reuse、restore 或 recompute 事件。
5. 所有输出都有 `diagnostic_only=true`。
6. quality 和 violation 保持 `risk_evidence_insufficient`，不宣称 tail SLO。
7. 如果 `full_lru` 仍然最快，或 TTFT 对 B 不敏感，就报告事件分解，不修改策略来制造预期排序。

### P3：测试和文档收尾

先运行相关测试：

```bash
pytest -q tests/test_qwen_session_backend.py \
  tests/test_policy_session_budget.py \
  tests/test_session_summary_visuals.py \
  tests/test_policy_quality_audit.py
```

再运行全量测试：

```bash
pytest -p no:cacheprovider -q \
  -k 'not test_qwen2_profile_measurement_appends_pythonpath'
```

已知例外是 `test_qwen2_profile_measurement_appends_pythonpath`，该测试硬编码了旧 worktree 路径，失败时不能直接判断为业务回归。

最后更新本交接文档，补充：

- OOM 根因和修复方式。
- 修复前后的显存表现。
- 9 个 batch 的覆盖结果。
- 四档 B 的实际数值。
- 五条策略的事件分解。
- 仍然属于 diagnostic evidence 的指标范围。

